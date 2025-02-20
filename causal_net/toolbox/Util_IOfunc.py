__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import time, os
#import ruamel.yaml  as yaml
import yaml
#import warnings
#warnings.simplefilter('ignore', yaml.error.MantissaNoDotYAML1_1Warning)

from pprint import pprint
import csv

#...!...!..................
def read_yaml(ymlFn,verb=1):
        if verb: print('  read  yaml:',ymlFn,end='')
        start = time.time()
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)

        ymlFd.close()
        if verb>1: print(' done  elaT=%.1f sec'%(time.time() - start))
        else: print()
        return bulk

#...!...!..................
def write_yaml(rec,ymlFn,verb=1):
        start = time.time()
        ymlFd = open(ymlFn, 'w')
        yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1024
        if verb:
                print('  closed  yaml:',ymlFn,' size=%.1f kB'%xx,'  elaT=%.1f sec'%(time.time() - start))

   
#...!...!..................
def read_one_csv(fname,delim=','):
    print('read_one_csv:',fname)
    tabL=[]
    with open(fname) as csvfile:
        drd = csv.DictReader(csvfile, delimiter=delim)
        print('see %d columns'%len(drd.fieldnames),drd.fieldnames)
        for row in drd:
            tabL.append(row)
            
        print('got %d rows \n'%(len(tabL)))
    #print('LAST:',row)
    return tabL,drd.fieldnames

#...!...!..................
def write_one_csv(fname,rowL,colNameL):
    print('write_one_csv:',fname)
    print('export %d columns'%len(colNameL), colNameL)
    with open(fname,'w') as fou:
        dw = csv.DictWriter(fou, fieldnames=colNameL)#, delimiter='\t'
        dw.writeheader()
        for row in rowL:
            dw.writerow(row)    


#...!...!..................
def expand_dash_list(inpL):
    # expand list if '-' are present
    outL=[]
    for x in inpL:
        if '-' not in x:
            outL.append(x) ; continue
        # must contain string: xxxx[n1-n2]
        ab,c=x.split(']')
        assert len(ab)>3
        a,b=ab.split('[')
        print('abc',a,b,c)
        nL=b.split('-')
        for i in range(int(nL[0]),int(nL[1])+1):
            outL.append('%s%d%s'%(a,i,c))
    print('EDL:',inpL,'  to ',outL)
    return outL
''' - - - - - - - - - 
Offset-aware time, usage:

*) get current date:
t1=time.localtime()   <type 'time.struct_time'>

*) convert to string: 
timeStr=dateT2Str(t1)

*) revert to struct_time
t2=dateStr2T(timeStr)

*) compute difference in sec:
t3=time.localtime()
delT=time.mktime(t3) - time.mktime(t1)
delSec=delT.total_seconds()
or delT is already in seconds
'''

#...!...!..................
def dateT2Str(xT):  # --> string
    nowStr=time.strftime("%Y%m%d_%H%M%S_%Z",xT)
    return nowStr

#...!...!..................
def dateStr2T(xS):  #  --> datetime
    yT = time.strptime(xS,"%Y%m%d_%H%M%S_%Z")
    return yT

