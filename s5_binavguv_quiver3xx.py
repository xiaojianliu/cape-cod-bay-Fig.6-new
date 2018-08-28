#http://www.ngdc.noaa.gov/mgg/coast/
# coast line data extractor

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:27:09 2012

@author: vsheremet
"""
from mpl_toolkits.basemap import Basemap  
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
import numpy as np
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
from matplotlib.dates import date2num,num2date
import numpy as np
#from pydap.client import open_url
import matplotlib.pyplot as plt
#from SeaHorseLib import *
#from datetime import *
#from scipy import interpolate
#import sys
#from SeaHorseTide import *
#import shutil
import matplotlib.mlab as mlab
import matplotlib.cm as cm
FNCL='necscoast_worldvec.dat'
# lon lat pairs
# segments separated by nans

CL=np.genfromtxt(FNCL,names=['lon','lat'])
################################################################
lonlat=[]
ud=[]
vd=[]
dr=np.load('dr.npy').tolist()

for a in np.arange(len(dr['lon'])):
    if "%s+%s"%(dr['lon'][a],dr['lat'][a]) not in lonlat:
        
        lonlat.append("%s+%s"%(dr['lon'][a],dr['lat'][a]))
        #lat.append(dr['lat'][a])
for a in np.arange(len(lonlat)):
    u=[]
    v=[]
    for b in np.arange(len(dr['lon'])):
        if "%s+%s"%(dr['lon'][b],dr['lat'][b])==lonlat[a]:
            u.append(dr['uudiff'][b])
            v.append(dr['vvdiff'][b])
    ud.append(u)
    vd.append(v)
            
umean=[]
vmean=[]
#speadmean=[]
ustd=[]
vstd=[]
#############################################################
lonlat1=[]
ud1=[]
vd1=[]
dr1=np.load('dr1.npy').tolist()
for a in np.arange(len(dr1['lon'])):
    if "%s+%s"%(dr1['lon'][a],dr1['lat'][a]) not in lonlat1:
        
        lonlat1.append("%s+%s"%(dr1['lon'][a],dr1['lat'][a]))
        #lat.append(dr['lat'][a])
for a in np.arange(len(lonlat1)):
    u1=[]
    v1=[]
    for b in np.arange(len(dr1['lon'])):
        if "%s+%s"%(dr1['lon'][b],dr1['lat'][b])==lonlat1[a]:
            u1.append(dr1['uudiff'][b])
            v1.append(dr1['vvdiff'][b])
    ud1.append(u1)
    vd1.append(v1)
            
umean=[]
vmean=[]
#speadmean=[]
ustd=[]
vstd=[]
##############################################################

fig,axes=plt.subplots(2,1,figsize=(15,22))
plt.subplots_adjust(wspace=0.1,hspace=0.1)
m = Basemap(projection='cyl',llcrnrlat=41.5,urcrnrlat=42.2,\
            llcrnrlon=-70.75,urcrnrlon=-69.8,resolution='h')#,fix_aspect=False)
    #  draw coastlines
m.drawcoastlines()
m.ax=axes[0]
m.fillcontinents(color='grey',alpha=1,zorder=2)
#m.drawmapboundary()
#draw major rivers
#m.drawrivers()
parallels = np.arange(41.5,42.2,0.1)
m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=15,zorder=0)
meridians = np.arange(-70.75,-69.8,0.1)
m.drawmeridians(meridians,labels=[0,0,0,1],dashes=[1,1000],fontsize=15,zorder=0)

#plt.plot(CL['lon'],CL['lat'])
for a in np.arange(len(ud)):
    
    axes[0].text(float(lonlat[a][0:7])-0.025,float(lonlat[a][8:14])+0.01,'(%s,'%(round(np.std(ud[a]),3)),fontsize=12)
    axes[0].text(float(lonlat[a][0:7])-0.025,float(lonlat[a][8:14])-0.01,'%s)'%(round(np.std(vd[a]),3)),fontsize=12)
    
    #*+str(round(np.mean(ud[a]),3)))

    axes[0].plot([float(lonlat[a][0:7])-0.025,float(lonlat[a][0:7])-0.025],[float(lonlat[0][8:14])+0.5,float(lonlat[-1][8:14])-0.5],color='black')

for a in np.arange(len(ud)):
    #axes.text(float(lonlat[a][0:7])-0.025,float(lonlat[a][8:14])+0.025,'mean')
    #*+str(round(np.mean(ud[a]),3)))

    axes[0].plot([float(lonlat[a][0:7])-1,float(lonlat[-1][0:7])+1],[float(lonlat[a][8:14])-0.025,float(lonlat[a][8:14])-0.025],color='black')

axes[0].plot([float(lonlat[a][0:7])-1,float(lonlat[-1][0:7])+1],[float(lonlat[0][8:14])+0.025,float(lonlat[0][8:14])+0.025],color='black')


axes[0].axis([-70.75,-69.8,41.5,42.2])

##################################################################

m1 = Basemap(projection='cyl',llcrnrlat=41.5,urcrnrlat=42.2,\
            llcrnrlon=-70.75,urcrnrlon=-69.8,resolution='h')#,fix_aspect=False)
    #  draw coastlines
m1.drawcoastlines()
m1.ax=axes[1]
m1.fillcontinents(color='grey',alpha=1,zorder=2)
#m.drawmapboundary()
#draw major rivers
#m.drawrivers()
parallels = np.arange(41.5,42.2,0.1)
m1.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=15,zorder=0)
meridians = np.arange(-70.75,-69.8,0.1)
m1.drawmeridians(meridians,labels=[0,0,0,1],dashes=[1,1000],fontsize=15,zorder=0)

#plt.plot(CL['lon'],CL['lat'])
for a in np.arange(len(ud)):
    
    axes[1].text(float(lonlat[a][0:7])-0.02,float(lonlat[a][8:14]),'%s'%(round(np.std(ud1[a]),3)),fontsize=12)
    
    #axes[1].text(float(lonlat[a][0:7])-0.025,float(lonlat[a][8:14])-0.01,',%s)'%(round(np.std(vd[a]),3)))
    
    #*+str(round(np.mean(ud[a]),3)))

    axes[1].plot([float(lonlat[a][0:7])-0.025,float(lonlat[a][0:7])-0.025],[float(lonlat[0][8:14])+0.5,float(lonlat[-1][8:14])-0.5],color='black')

for a in np.arange(len(ud)):
    #axes.text(float(lonlat[a][0:7])-0.025,float(lonlat[a][8:14])+0.025,'mean')
    #*+str(round(np.mean(ud[a]),3)))

    axes[1].plot([float(lonlat[a][0:7])-1,float(lonlat[-1][0:7])+1],[float(lonlat[a][8:14])-0.025,float(lonlat[a][8:14])-0.025],color='black')

axes[1].plot([float(lonlat[a][0:7])-1,float(lonlat[-1][0:7])+1],[float(lonlat[0][8:14])+0.025,float(lonlat[0][8:14])+0.025],color='black')


axes[1].axis([-70.75,-69.8,41.5,42.2])
axes[0].set_title('a std of difference in u,v',fontsize=20)
axes[1].set_title('b std of difference in speed',fontsize=20)
plt.savefig('uvspeed.eps',format='eps',dpi=300,bbox_inches='tight')


u_std_mean=np.std(dr['uudiff'])

v_std_mean=np.std(dr['vvdiff'])

speed_std_mean=np.std(dr1['uudiff'])

print u_std_mean
print v_std_mean
print speed_std_mean