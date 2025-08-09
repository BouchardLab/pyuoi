#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

''' = = = = =  NPZ advanced storage = = =
It can hold:
* python dictionaries which must pass: json.dumps(dict)
* single float or int variables w/o np-array packing. It is recovered as 1-value array
* arbitrary numpy array (a large size payloads)
   - for an array of arbitrary strings must declare  dtype='object' at write and use .decode("utf-8")  to unpack 
'''

import numpy as np
import time, os
import json
from pprint import pprint

#...!...!..................
def write_data_npz(dataD,outF,metaD=None,verb=1):
    assert type(dataD)!=type(None)
    assert len(outF)>0
    
    if metaD!=None:
        metaJ=json.dumps(metaD, default=str)
        dataD['meta.JSON']=metaJ
    
    if verb>1:
            print('saving data as npz:',outF)
    start = time.time()
    
    # Prepare data for npz saving
    npz_data = {}
    for item in dataD:
        rec = dataD[item]
        if verb>1: print('x=',item,type(rec))
        
        if isinstance(rec, dict):
            # Serialize nested dictionary to a JSON string
            rec_str = json.dumps(rec, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
            rec = np.array([rec_str], dtype='object')
        elif type(rec)==str:
            rec = np.array([rec], dtype='object')
        elif type(rec)!=np.ndarray:
            rec = np.array([rec])
        
        npz_data[item] = rec
    
    # Save using np.savez_compressed for better compression
    np.savez_compressed(outF, **npz_data)
    
    xx = os.path.getsize(outF)/1048576
    print('closed  npz:',outF,' size=%.2f MB, elaT=%.1f sec'%(xx,(time.time() - start)))

    
#...!...!..................
def read_data_npz(inpF,verb=1):
    if verb>1:
            print('read data from npz:',inpF)
    start = time.time()
    
    # Load npz file
    npz_file = np.load(inpF, allow_pickle=True)
    objD = {}
    
    for x in npz_file.files:
        if verb>1: print('\nitem=',x,type(npz_file[x]),npz_file[x].shape,npz_file[x].dtype)
        
        if npz_file[x].dtype==object:
            obj = npz_file[x]
            if verb>0: print('read str:',x,len(obj),type(obj))
        else:
            obj = npz_file[x]
            if verb>0: print('read obj:',x,obj.shape,obj.dtype)
        objD[x] = obj
    
    # Close the npz file
    npz_file.close()
    
    # Extract metadata if present
    try:
        inpMD = json.loads(objD.pop('meta.JSON')[0])
        if verb>1: print('  recovered meta-data with %d keys'%len(inpMD))
    except:
        inpMD = None
    
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
    
    # this works too??:
    # three=np.array(['record aaaa','much longer record bbb'], dtype='object')
    
    text='This is text1'  

    metaD={"age":17,"dom":"white","dates":[11,22,33]}
   
    outD={'one':one,'two':two,'var1':var1,'atext':text,'three':three}

    # ... nested dict of numpy
    subD={'one1':one,'two1':two}
    outD['sub']=subD
    
    write_data_npz(outD,outF,metaD=metaD,verb=verb)

    print('\nM: *****  verify by reading it back from',outF)
    big,meta2=read_data_npz(outF,verb=verb)
    from pprint import pprint        
    print(' recovered meta-data'); pprint(meta2)
    print('dump read-in data')
    for key, item in big.items():
        # Detect nested dict stored as JSON in object array
        if isinstance(item, np.ndarray) and item.dtype == object and len(item) == 1 and isinstance(item[0], str):
            try:
                parsed = json.loads(item[0])
                if isinstance(parsed, dict):
                    for subk, subv in parsed.items():
                        print(f"{key}.{subk}: {subv}")
                    continue
            except json.JSONDecodeError:
                pass
        # Fallback: print numpy arrays or other items
        if isinstance(item, np.ndarray):
            print(f"{key}: {item.tolist()}")
        else:
            print(f"{key}: {item}")
  
    #decode one string from string-array
    rec2=big['three'][1]  # No need for .decode("utf-8") in npz
    print('rec2:',type(rec2),rec2)
    print('\n check raw content:   python -c "import numpy as np; data=np.load(\'%s\', allow_pickle=True); print(data.files); [print(k, data[k].shape, data[k].dtype) for k in data.files]"\n'%outF) 
