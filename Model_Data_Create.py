#!/usr/bin/env python
import os,pdb,h5py
import numpy as np
from optparse import OptionParser
import scipy.io as sio


"""
Authors : Alex Bujan (adapted from Kris Bouchard)
Date    : 08/12/2015
"""

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("--mdlW",type="string",default="ExpI",\
        help="weight distribution (options: 'Gaus','ExpI','Clst1','Clst2','Lap','Uni','ID')")
    parser.add_option("--mdlsz",type="int",default=100,\
        help="number of non-null dimensions in the model")
    parser.add_option("--v1",type="float",default=.0,\
        help="magnitude of the noise in the model")
    parser.add_option("--v2",type="float",default=4.,\
        help="ratio of null/non-null dimensions in the model")
    parser.add_option("--v3",type="float",default=3,\
        help="ratio of samples/parameters in the model")
    parser.add_option("--store",action="store_true",dest="store",default=True,\
        help="store results to file")
    parser.add_option("--saveAs",type="string",default='hdf5',\
        help="File format to store the data. Options: hdf5(default), mat, txt")
    parser.add_option("--path",type="string",default=os.getcwd(),\
        help="path to store the results (default: current directory)")
    parser.add_option("--seed",type="int",default=np.random.randint(9999),\
        help="seed for generating pseudo-random numbers (default: random seed)")
    parser.add_option("--dtype",type="string",default='f4',\
        help="float type to be used. Options: 'f4' and 'f8' bit")


    (options, args) = parser.parse_args()

    Model_Data_Create(mdlW=options.mdlW,v1=options.v1,\
                    v2=options.v2,v3=options.v3,\
                    mdlsz=options.mdlsz,store=options.store,\
                    path=options.path,seed=options.seed,\
                    saveAs=options.saveAs,dtype=options.dtype)

def Model_Data_Create(mdlW,v1,v2,v3,mdlsz=100,seed=np.random.randint(9999),\
                        store=False,path=os.getcwd(),saveAs='hdf5',\
                        dtype='f4'):

    """
    Model_Data_Create
    -----------------

    Creates data samples using different underlying coefficient distributions.

    Input:
        -mdlW       : weight distribution; options: 'Gaus','ExpI'(default),
                        'Clst1','Clst2','Lap','Uni','ID'
        -mdlsz      : number of non-null dimensions in the model
        -v1         : magnitude of the noise in the model
        -v2         : ratio of null/non-null dimensions in the model
        -v3         : ratio of samples/parameters in the model
        -store      : store results to file
        -saveAs     : file format to store the data; options: hdf5(default),
                        mat, txt
        -path       : path to store the results (default: current directory)
        -seed       : seed for generating pseudo-random numbers (default: 
                        a different random seed is generated every time

    Output:
        - X         : design matrix
        - y         : response
        - Wact      : true weights
    """
    np.random.seed(seed)
    '''
    #create model weights
    #--------------------
    '''
    if mdlW=='Gaus':
        mn =-4
        mx = 4
        W  = (mx-mn)*np.random.normal(size=(mdlsz,1))
    elif mdlW=='Uni':
        mn =-5
        mx = 5
        W  = mn+(mx-mn)*np.random.uniform(size=(mdlsz,1))
    elif mdlW=='Lap':
        mn = -2
        mx = 2
        W  = np.exp(np.linspace(mn,mx,np.floor(mdlsz/2.)))
        W  = np.hstack((-1*W,W))[:,np.newaxis]
    elif mdlW=='ExpI':
        mn = -2
        mx = 2
        if mdlsz==1:
            W  = np.exp(np.linspace(mn,mx,1))
        else:
            W  = np.exp(np.linspace(mn,mx,np.floor(mdlsz/2.)))
            W1 = W-np.max(W)-.1*np.max(W)
            W2 = np.abs(W-np.max(W)-.1*np.max(W))
            W  = np.hstack((W1,W2))[:,np.newaxis]
    elif mdlW=='Clst1':
        mn = 5
        mx = 30
        lp = np.linspace(mn,mx,6)
        for i in range(5):
            lpW = lp[i]+np.random.uniform(size=(np.floor(mdlsz/5.),1))
            if i==0:
                W = lpW
            else:
                W = np.vstack((W,lpW))
    elif mdlW=='Clst2':
        mn = 3
        mx = 20
        lp = np.linspace(mn,mx,6)
        for i in range(5):
            lpW = np.exp(np.linspace(lp[i],lp[i+1],np.floor(mdlsz/10.)))[:,np.newaxis]
            if i==0:
                W = np.vstack((-lpW,lpW))
            else:
                W = np.vstack((W,-lpW,lpW))
    elif mdlW=='ID':
        W = 10*np.ones(size=(mdlsz,1))

    #total number of data samples
    nd = int(np.floor(v3*(mdlsz+mdlsz*v2)))
    '''
    #generate input data
    #-------------------
    '''
    #data for non-null dimensions
    Dat     = 3*np.random.normal(size=(mdlsz,nd))
    #data for null dimensions
    Dat2    = 3*np.random.normal(size=(int(1+round(v2*mdlsz)),nd))
    #design matrix // input data // non-null and null dimensions
    DDat    = np.vstack((Dat,Dat2))
    '''
    #ground truth
    #------------
    '''
    #dim(Wact)<-mdlsz+mdlsz*v2+1
    tmp = np.zeros((int(1+round(v2*mdlsz)),1))
    Wact    = np.vstack((W,tmp))
    '''
    #output data
    #-----------
    '''
    #y <- output // dim(y) <- nd
    y = np.dot(W.T,Dat)+v1*np.sum(np.abs(W))*np.random.normal(size=nd)
    y-=np.mean(y)

    if store:
        name = '%s_%s_%s_%s_%s_%s'%(mdlW,mdlsz,str(v1*100),str(v2*100),str(v3*100),dtype)
        if saveAs=='hdf5':
            with h5py.File('%s/Model_Data_%s.h5'%(path,name),'w') as f:
                f.attrs['mdlW']   = mdlW
                f.attrs['mdlsz']     = mdlsz
                f.attrs['v1']  = v1
                f.attrs['v2']  = v2
                f.attrs['v3'] = v3
                f.attrs['seed']   = seed
                g = f.create_group('data')
                g.create_dataset(name='X',data=DDat.T,dtype=dtype,\
                                shape=DDat.T.shape,compression="gzip")
                g.create_dataset(name='y',data=np.ravel(y),dtype=dtype,\
                                shape=np.ravel(y).shape,compression="gzip")
                g.create_dataset(name='Wact',data=Wact,dtype=dtype,\
                                shape=Wact.shape,compression="gzip")
        elif saveAs=='txt':
            np.savetxt('%s/X_%s.txt'%(path,name),DDat.T.astype(dtype))
            np.savetxt('%s/y_%s.txt'%(path,name),np.ravel(y).astype(dtype))
            np.savetxt('%s/Wact_%s.txt'%(path,name),Wact.astype(dtype))

        elif saveAs=='mat':
            sio.savemat('%s/Model_Data_%s.mat'%(path,name),\
                        {'X'    : DDat.T.astype(dtype),\
                         'y'    : np.ravel(y).astype(dtype),\
                         'Wact' : Wact.astype(dtype)})

        print('\nData Model:')
        print('\t* No covariates:\t%i'%DDat.shape[0])
        print('\t* No samples   :\t%i'%DDat.shape[1])
        print('Data stored in %s'%path)
    else:
        return DDat.T,np.ravel(y),Wact

if __name__=='__main__':
    main()
