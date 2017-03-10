# for batch run, use sbatch runuoi.slrm
# for interactive run:
# comment out the definition of conf and sc in this file (don't forget to add
# back when doing batch runs), then
# salloc -N 2 -p debug -t 30 --ccm --qos=premium
# then
# bash
# module load spark (or spark/1.5.1)
# start-all.sh
# pyspark --master $SPARKURL --num-executors 10  --driver-memory 1G --executor-memory 1G 
# and use execfile('test.py') to load and run

import numpy as np
import itertools
from pyspark import SparkContext
from sklearn import linear_model as lm
from sklearn.cross_validation import KFold
from pyspark import SparkConf
from h5py import File

# comment out the following two lines if you use pyspark interactively
#conf = SparkConf()
#sc = SparkContext(appName="UoItest", conf=conf)

numbootstraps = 5
trainfraction = .8

lambda_vals = [0.01, 0.1, 1.] 
numlambdas = len(lambda_vals)
fname = "test.h5"
numcvsplits = 5

numexamples, n = File(fname, "r")["data/X"].shape
cvFoldsGenerator = KFold(numexamples, n_folds = numcvsplits, shuffle=True)
cvFolds = [pair for pair in cvFoldsGenerator]

def intersection_of_supports(paramlist, coeffsList):
    """Compute intersection of supports across bootstraps for each lambda, here
    paramlist is a list of (lambda, bootstrap_index) values and the corresponding row
    of coeffsList contains the corresponding coefficients"""
    supports = np.ones((numlambdas, n))*np.nan
    for lambdaIdx in xrange(numlambdas):
        relevantRowIndices = [idx for idx in xrange(len(paramlist)) if paramlist[idx][1] == lambda_vals[lambdaIdx]]
        intersection = np.where( coeffsList[relevantRowIndices[0]] != 0)[0]
        for rowIdx in relevantRowIndices:
            intersection = np.intersect1d(intersection, np.where(coeffsList[rowIdx] != 0)[0]).astype('int')
        supports[lambdaIdx, :len(intersection)] = intersection
    return supports

def compute_lasso(X_train,y_train,lambda_val):
    """Compute Lasso regression for given lambda and bootstrap sample"""
    outLas = lm.Lasso(alpha=lambda_val)
    outLas.fit(X_train,\
               y_train-y_train.mean())
    coeffs = outLas.coef_
    return coeffs

def lasso_wrapper(pair):
    """Takes (boostrap_seed, lambda_val), and calls compute lasso with appropriate parameters;
    assume X and y are stored locally on each node"""
    bootstrap_seed, lambda_val = pair
    np.random.seed(bootstrap_seed)
    fin = File(fname, "r")
    numtrain = int(np.ceil(trainfraction*numexamples))
    trainIndices = np.random.permutation(numexamples)[:numtrain]
    X_train = fin["data/X"].value[trainIndices]
    y_train = fin["data/y"].value[trainIndices]
    return compute_lasso(X_train, y_train, lambda_val)

def compute_OLS_with_support(X_train,X_test,y_train,y_test,support):
    """Estimate model parameters using linear regression with support for each
        lambda and bootstrap sample"""
    #train
    outLR = lm.LinearRegression()
    idx = support[~np.isnan(support)].astype('int')
    outLR.fit(X_train[:,idx],\
              y_train-y_train.mean())
    parameter_values = outLR.coef_
    coeffs = np.zeros(n)
    coeffs[idx] = parameter_values
    #test
    yhat = X_test.dot(coeffs)
    r = np.corrcoef(yhat,y_test-y_test.mean())
    R2 = r[1,0]**2
    return coeffs,R2

def ols_wrapper(triple):
    """Takes (bootstrap_seed, support, cvFoldIndex) and calls compute_OLS_with_support with appropriate parameters;
    assumes X and y are stored locally on each node; cvFolds[cvFoldIndex] = (trainAndValidationIndices, testIndices)"""
    bootstrap_seed, support, cvFoldIndex = triple
    trainAndValidationIndices, testIndices = cvFolds[cvFoldIndex]
    support = support[~np.isnan(support)]
    np.random.seed(bootstrap_seed)
    fin = File(fname, "r")
    numtrain = int(np.ceil(trainfraction*len(trainAndValidationIndices)))
    trainIndices = trainAndValidationIndices[:numtrain]
    validationIndices = trainAndValidationIndices[numtrain:]
    print validationIndices
    X = fin["data/X"].value
    y = fin["data/y"].value
    X_train = X[trainIndices]
    y_train = y[trainIndices]
    X_validation = X[validationIndices]
    y_validation = y[validationIndices]
    return compute_OLS_with_support(X_train, X_validation, y_train, y_validation, support)

def union_of_intersections(paramlist, coeffsAndR2List):
    """ Compute the "median model" across bootstrap samples for the best lambda
        value"""
    coeffs = [pair[0] for pair in coeffsAndR2List]
    R2 = [pair[1] for pair in coeffsAndR2List]

    bootstrapSeeds = np.unique([triple[0] for triple in paramlist])
    cvFoldIndices = np.unique([triple[2] for triple in paramlist])

    bootstrapBestCoeffs = np.zeros((numcvsplits, numbootstraps, n))
    for cvIdx in cvFoldIndices:
        for seedIdx, seedVal in enumerate(bootstrapSeeds):
            relevantRowIndices = [idx for idx in xrange(len(paramlist)) if (paramlist[idx][0] == seedVal and paramlist[idx][2] == cvIdx)]    
            bestModelIdx = relevantRowIndices[0]
            for rowIdx in relevantRowIndices:
                if R2[rowIdx] <= R2[bestModelIdx]:
                    bestModelIdx = rowIdx
                    bootstrapBestCoeffs[cvIdx, seedIdx, :] = coeffs[bestModelIdx]
    return np.median(bootstrapBestCoeffs, 1)

def test_UoI(X_test,y_test,coeffs):
    """Test UoI powered models on held out data"""
    yhat = X_test.dot(coeffs)
    r = np.corrcoef(yhat,y_test-y_test.mean())
    R2  = r[1,0]**2
    rsd = (y_test-y_test.mean())-yhat
    bic = (m-m_frac)*np.log(np.sum(rsd**2)*1./(m-m_frac))+\
               np.log(m-m_frac)*n
    return R2,rsd,bic

firstBootstrapSeeds  = np.random.randint(9999,size=numbootstraps)
paramlist = [item for item in  itertools.product(firstBootstrapSeeds, lambda_vals)]
paramrdd = sc.parallelize(paramlist, numSlices=len(paramlist))
coeffsList = paramrdd.map(lasso_wrapper).collect()
supports = intersection_of_supports(paramlist, coeffsList)

secondBootstrapSeeds = np.random.randint(9999,size=numbootstraps)
cvFoldIndices = np.arange(len(cvFolds), dtype='int')
paramlist = [item for item in itertools.product(secondBootstrapSeeds, supports, cvFoldIndices)]
paramrdd  = sc.parallelize(paramlist, numSlices=len(paramlist))
coeffsAndR2List = paramrdd.map(ols_wrapper).collect()
bestCoeffs = union_of_intersections(paramlist, coeffsAndR2List)
