#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

''' = = = = =  HD5 advanced storage = = =
It can hold:
* python dictionaries which must pass: json.dumps(dict)
* single float or int variables w/o np-array packing. It is recovered as 1-value array
* arbitrary numpy array (a large size payloads)
   - for an array of arbitrary strings must declare  dtype='object' at write and use .decode("utf-8")  to unpack 
'''

import numpy as np
import h5py, time, os
import json,time
from pprint import pprint

#...!...!..................
def write4_data_hdf5(dataD,outF,metaD=None,verb=1):
    assert type(dataD)!=type(None)
    assert len(outF)>0
    
    if metaD!=None:
        #pprint(metaD)
        metaJ=json.dumps(metaD, default=str)
        #print('meta.JSON:',metaJ)
        dataD['meta.JSON']=metaJ
    
    dtvs = h5py.special_dtype(vlen=str)
    h5f = h5py.File(outF, 'w')
    if verb>0:
            print('saving data as hdf5:',outF)
            start = time.time()
    for item in dataD:
        rec=dataD[item]
        if verb>1: print('x=',item,type(rec))
        if type(rec)==str: # special case
            dset = h5f.create_dataset(item, (1,), dtype=dtvs)
            dset[0]=rec
            if verb>0:print('h5-write :',item, 'as string',dset.shape,dset.dtype)
            continue
        if type(rec)!=np.ndarray: # packs a single value into np-array
            rec=np.array([rec])

        h5f.create_dataset(item, data=rec)
        if verb>0:print('h5-write :',item, rec.shape,rec.dtype)
    h5f.close()
    xx=os.path.getsize(outF)/1048576
    print('closed  hdf5:',outF,' size=%.2f MB, elaT=%.1f sec'%(xx,(time.time() - start)))

    
#...!...!..................
def read4_data_hdf5(inpF,verb=1):
    if verb>0:
            print('read data from hdf5:',inpF)
            start = time.time()
    h5f = h5py.File(inpF, 'r')
    objD={}
    for x in h5f.keys():
        if verb>1: print('\nitem=',x,type(h5f[x]),h5f[x].shape,h5f[x].dtype)
        #if x in ['calTag','dataFile','date'] : continue
        if h5f[x].dtype==object:
            obj=h5f[x][:]
            #print('bbb',type(obj),obj.dtype)
            if verb>0: print('read str:',x,len(obj),type(obj))
        else:
            obj=h5f[x][:]
            if verb>0: print('read obj:',x,obj.shape,obj.dtype)
        objD[x]=obj
    try:
        inpMD=json.loads(objD.pop('meta.JSON')[0])
        if verb>1: print('  recovered meta-data with %d keys'%len(inpMD))
    except:
        inpMD=None
    if verb>0:
        print(' done h5, num rec:%d  elaT=%.1f sec'%(len(objD),(time.time() - start)))

    h5f.close()
    return objD,inpMD



#=================================
#=================================
#   U N I T   T E S T
#=================================
#=================================

if __name__=="__main__":
    print('testing h5IO ver 3')
    outF='abcTest.h5'
    verb=1
    
    var1=float(15) # single variable
    one=np.zeros(shape=5,dtype=np.int16); one[3]=3
    two=np.zeros(shape=(2,3)); two[1,2]=4

    three=np.empty((2), dtype='object')
    three[0]='record aaaa'
    three[1]='much longer record bbb'
    # WARN:  all decalred elements of three[] must be initialized before writeing H5
    
    # this works too:
    # three=np.array(['record aaaa','much longer record bbb'], dtype='object')
    
    text='This is text1'  

    metaD={"age":17,"dom":"white","dates":[11,22,33]}
   
    outD={'one':one,'two':two,'var1':var1,'atext':text,'three':three}

    write4_data_hdf5(outD,outF,metaD=metaD,verb=verb)

    print('\nM: *****  verify by reading it back from',outF)
    big,meta2=read4_data_hdf5(outF,verb=verb)
    from pprint import pprint        
    print(' recovered meta-data'); pprint(meta2)
    print('dump read-in data')
    for x in big:
        print('\nkey=',x); pprint(big[x])
  
    #decode one string from string-array
    rec2=big['three'][1].decode("utf-8") 
    print('rec2:',type(rec2),rec2)
    print('\n check raw content:   h5dump %s\n'%outF)
