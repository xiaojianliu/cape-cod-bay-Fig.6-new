# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 00:50:48 2017

@author: xiaojian
"""
import numpy as np
import datetime as dt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import csv
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from math import sqrt
from netCDF4 import Dataset

#plt.axis([-70.94,-70.0,41.52659,42.562711])
lat=np.linspace(41.8,42.4,10)
lon=np.linspace(-70.15,-70.75,10)
lo=[]
la=[]
for a in np.arange(len(lon)):
    for b in np.arange(len(lat)):
        lo.append(lon[a])
        la.append(lat[b])

tt2012=np.load('tt2012.npy')
h2012=np.load('h2012.npy')
fig=plt.figure(figsize=(15,10))
axes2=fig.add_subplot(2,2,1)
axes3=fig.add_subplot(2,2,2)

#axes3=fig.add_subplot(2,2,3)

dian=['a','b','c','d','e','f','g','h','i','j']

FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])
#axes1.plot(CL['lon'],CL['lat'])
#axes1.axis([-70.94,-70.0,41.52659,42.562711])
#axes1.scatter(lon,lat,s=5,color='red')
#for a in np.arange(len(lon)):
#    axes1.text(lon[a]-0.05,lat[a],dian[a])
#axes1.xaxis.tick_top() 
#axes1.set_xlabel('a')
axes2.set_xlabel('a')
axes3.set_xlabel('b')
axes2.set_title('2012')
axes3.set_title('2013')

time2012=np.load('time2012.npy')
temp2012=np.load('temp2012.npy')
ttt1=[]
for a in np.arange(len(time2012)):
    
    ttt1.append('''%s/%s'''%(time2012[a].month,time2012[a].day))
for a in np.arange(len(temp2012)):
    data2010s1=pd.Series(temp2012[a],index=list(ttt1))
    data2010s1.plot(linestyle='-',ax=axes2,label=dian[a])

###########################################################
###########################################################
time2013=np.load('time2013.npy')
temp2013=np.load('temp2013.npy')
ttt=[]
for a in np.arange(len(time2013)):
    
    ttt.append('''%s/%s'''%(time2013[a].month,time2013[a].day))
for a in np.arange(len(temp2013)):
    data2010s=pd.Series(temp2013[a],index=list(ttt))
    data2010s.plot(linestyle='-',ax=axes3,label=dian[a])

axes2.legend()
axes2.set_ylim([0,14])
#axes[1].plot([time2013[0],time2013[-1]],[10,10],color='black')
#axes[1].set_ylabel('number')
axes2.set_ylabel('temperature(Degrees Celsius)')
############################################################
axes3.legend()
axes3.set_ylim([0,14])
#axes[1].plot([time2013[0],time2013[-1]],[10,10],color='black')
#axes[1].set_ylabel('number')
axes3.set_ylabel('temperature(Degrees Celsius)')

plt.savefig('zx20131',dpi=300)