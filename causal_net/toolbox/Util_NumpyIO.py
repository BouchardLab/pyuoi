#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

''' = = = = =  Numpy NPZ advanced storage = = =
It can hold:
* python dictionaries which must pass: json.dumps(dict)
* single float or int variables w/o np-array packing. It is recovered as 1-value array
* arbitrary numpy array (a large size payloads)
   - for an array of arbitrary strings must declare  dtype='object' at write and use .decode("utf-8")  to unpack 
'''

import numpy as np
import time, os
import json,time
from pprint import pprint

#...!...!..................
def write_data_npz(dataD,outF,metaD=None,verb=1):
    assert type(dataD)!=type(None)
    assert len(outF)>0
    start = time.time()
    
    # Create a copy to avoid modifying original data
    saveD = dataD.copy()
    
    if metaD!=None:
        #pprint(metaD)
        metaJ=json.dumps(metaD, default=str)
        #print('meta.JSON:',metaJ)
        saveD['meta.JSON']=np.array([metaJ], dtype='object')
    
    if verb>0:
            print('saving data as npz:',outF)
    
    # Process data to ensure all items are numpy arrays
    for item in list(saveD.keys()):
        rec=saveD[item]
        if verb>1: print('x=',item,type(rec))
        if type(rec)==str: # special case - convert string to object array
            saveD[item] = np.array([rec], dtype='object')
            if verb>0:print('npz-write :',item, 'as string array',saveD[item].shape,saveD[item].dtype)
            continue
        if type(rec)!=np.ndarray: # packs a single value into np-array
            saveD[item]=np.array([rec])
            if verb>0:print('npz-write :',item, saveD[item].shape,saveD[item].dtype)
        else:
            if verb>0:print('npz-write :',item, rec.shape,rec.dtype)

    # Save to NPZ format
    np.savez_compressed(outF, **saveD)
    
    xx=os.path.getsize(outF)/1048576
    if verb>0:
        print('closed  npz:',outF,' size=%.2f MB, elaT=%.1f sec'%(xx,(time.time() - start)))

    
#...!...!..................
def read_data_npz(inpF,verb=1):
    if verb>0:
            print('read data from npz:',inpF)
            start = time.time()
    
    npzData = np.load(inpF, allow_pickle=True)
    objD={}
    
    for x in npzData.files:
        obj = npzData[x]
        if verb>1: print('\nitem=',x,type(obj),obj.shape,obj.dtype)
        
        if obj.dtype==object:
            if verb>0: print('read obj:',x,len(obj),type(obj))
        else:
            if verb>0: print('read obj:',x,obj.shape,obj.dtype)
        objD[x]=obj
    
    npzData.close()
    
    try:
        inpMD=json.loads(objD.pop('meta.JSON')[0])
        if verb>1: print('  recovered meta-data with %d keys'%len(inpMD))
    except:
        inpMD=None
    if verb>0:
        print(' done npz, num rec:%d  elaT=%.1f sec'%(len(objD),(time.time() - start)))

    return objD,inpMD



#=================================
#=================================
#   U N I T   T E S T
#=================================
#=================================

if __name__=="__main__":
    print('testing npzIO ver 1')
    outF='abcTest.npz'
    verb=1
    
    var1=float(15) # single variable
    one=np.zeros(shape=5,dtype=np.int16); one[3]=3
    two=np.zeros(shape=(2,3)); two[1,2]=4

    three=np.empty((2), dtype='object')
    three[0]='record aaaa'
    three[1]='much longer record bbb'
    # WARN:  all decalred elements of three[] must be initialized before writeing NPZ
    
    # this works too:
    # three=np.array(['record aaaa','much longer record bbb'], dtype='object')
    
    text='This is text1'  

    metaD={"age":17,"dom":"white","dates":[11,22,33]}
   
    outD={'one':one,'two':two,'var1':var1,'atext':text,'three':three}

    write_data_npz(outD,outF,metaD=metaD,verb=verb)

    print('\nM: *****  verify by reading it back from',outF)
    big,meta2=read_data_npz(outF,verb=verb)
    from pprint import pprint        
    print(' recovered meta-data'); pprint(meta2)
    print('dump read-in data')
    for x in big:
        print('\nkey=',x); pprint(big[x])
  
    #get one string from string-array
    rec2=big['three'][1] 
    print('rec2:',type(rec2),rec2)
    print('\n check raw content with: python -c "import numpy as np; data=np.load(\'%s\', allow_pickle=True); print(list(data.files)); data.close()"\n'%outF)
