# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 02:37:22 2016

@author: Administrator
"""
import csv
from datetime import datetime
import datetime as dt
import pytz
import numpy as np
from matplotlib.path import Path
import pandas as pd
import matplotlib.pyplot as plt
FList = np.genfromtxt('data/FList.csv',dtype=None,names=['FNs'],delimiter=',',skip_header=1)
FList1=[]
time0=datetime(1858,11,17,00,00,00)
A=0
for Aa in np.arange(len(FList['FNs'])):
    #times=[]
    #hours=[]
    print Aa
    drifters = np.genfromtxt('''data/'''+FList['FNs'][Aa],dtype=None,names=['ids','start_time','lon','lat','depth'],delimiter=',',skip_header=1)  

    #drifters = np.genfromtxt('drifters.csv',dtype=None,names=['ids','start_time','lat','lon','depth'],delimiter=',',skip_header=2)    
    #a='''http://comet.nefsc.noaa.gov/erddap/tabledap/drifters.csv?id,time,latitude,longitude,depth,sea_water_temperature&time%3E=2011-12-10T00:00:00Z&time%3C=2014-01-11T00:00:00Z&latitude%3E=41.0&latitude%3C=43.0&longitude%3E=-70.0&longitude%3C=-67.0&depth%3E=-1&depth%3C=0'''
    #df=pd.read_csv(a,skiprows=[1])
    
    
    d=dict(ids=[],time=[],lat=[],lon=[],deep=[])
    try:
        for a in np.arange(len(drifters['ids'])):
            if abs(drifters['depth'][a])<=1:
                d['ids'].append(drifters['ids'][a])
                d['time'].append(drifters['start_time'][a])
                d['lat'].append(drifters['lat'][a])
                d['lon'].append(drifters['lon'][a])
                d['deep'].append(drifters['depth'][a])
                
        n=[]
        n.append(0)
        for a in np.arange(len(d['ids'])-1):
            if d['ids'][a]!=d['ids'][a+1] or ((datetime.strptime(d['time'][a+1], '%Y-%m-%d'+'T'+'%H:%M:%SZ')-datetime.strptime(d['time'][a], '%Y-%m-%d'+'T'+'%H:%M:%SZ')).days*24+(datetime.strptime(d['time'][a+1], '%Y-%m-%d'+'T'+'%H:%M:%SZ')-datetime.strptime(d['time'][a], '%Y-%m-%d'+'T'+'%H:%M:%SZ')).seconds/float(3600))>2:
                n.append(a+1)
        n.append(len(d['ids']))
        f='julu.csv'
        csvfile1 = file(f, 'wb')
        for a in np.arange(len(n)-1):
            drift=dict(ids=[],time=[],lon=[],lat=[],deep=[])
            for b in np.arange(len(d['ids'])):
                if b>=n[a] and b<n[a+1]:
                    drift['ids'].append(d['ids'][b])
                    drift['time'].append(d['time'][b])
                    drift['deep'].append(-abs(d['deep'][b]))
                    drift['lon'].append(d['lon'][b])
                    drift['lat'].append(d['lat'][b])
            drifter_data=[]
            drifter_data.append(drift['ids'])
            drifter_data.append(drift['time'])
            drifter_data.append(drift['lon'])
            drifter_data.append(drift['lat'])
            drifter_data.append(drift['deep'])
            dr=map(list, zip(*drifter_data))
            FN2='''d/'''+str(A)+'.csv'
            writer1 = csv.writer(csvfile1)
            #f.writelines(str(drift['ids'][0])+'_'+str(a)+'.csv')
            writer1.writerows([str(a)+'.csv'])
            csvfile = file(FN2, 'wb')
            writer = csv.writer(csvfile)
            writer.writerow(['ids', 'time', 'lon','lat','depth'])
            writer.writerows(dr)
            csvfile.close() 
            A=A+1
        csvfile1.close()
    except:
        pass
