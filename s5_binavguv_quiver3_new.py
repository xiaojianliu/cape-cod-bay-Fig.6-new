#http://www.ngdc.noaa.gov/mgg/coast/
# coast line data extractor

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:27:09 2012

@author: vsheremet
"""
from scipy import interpolate 
from mpl_toolkits.basemap import Basemap  
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
def sh_bindata(x, y, z, xbins, ybins):
    """
    Bin irregularly spaced data on a rectangular grid.

    """
    ix=np.digitize(x,xbins)
    iy=np.digitize(y,ybins)
    xb=0.5*(xbins[:-1]+xbins[1:]) # bin x centers
    yb=0.5*(ybins[:-1]+ybins[1:]) # bin y centers
    zb_mean=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_median=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_std=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_num=np.zeros((len(xbins)-1,len(ybins)-1),dtype=int)    
    for iix in range(1,len(xbins)):
        for iiy in range(1,len(ybins)):
#            k=np.where((ix==iix) and (iy==iiy)) # wrong syntax
            k,=np.where((ix==iix) & (iy==iiy))
            zb_mean[iix-1,iiy-1]=np.mean(z[k])
            zb_median[iix-1,iiy-1]=np.median(z[k])
            zb_std[iix-1,iiy-1]=np.std(z[k])
            zb_num[iix-1,iiy-1]=len(z[k])
            
    return xb,yb,zb_mean,zb_median,zb_std,zb_num
"""
from netCDF4 import Dataset

# read in etopo5 topography/bathymetry.
url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
etopodata = Dataset(url)

topoin = etopodata.variables['ROSE'][:]
lons = etopodata.variables['ETOPO05_X'][:]
lats = etopodata.variables['ETOPO05_Y'][:]
# shift data so lons go from -180 to 180 instead of 20 to 380.
topoin,lons = shiftgrid(180.,topoin,lons,start=False)
"""



"""
BATHY=np.genfromtxt('necscoast_noaa.dat',dtype=None,names=['coast_lon', 'coast_lat'])
coast_lon=BATHY['coast_lon']
coast_lat=BATHY['coast_lat']
"""

#BATHY=np.genfromtxt('coastlineNE.dat',names=['coast_lon', 'coast_lat'],dtype=None,comments='>')
#coast_lon=BATHY['coast_lon']
#coast_lat=BATHY['coast_lat']


# www.ngdc.noaa.gov
# world vector shoreline ascii
FNCL='necscoast_worldvec.dat'
# lon lat pairs
# segments separated by nans

CL=np.genfromtxt(FNCL,names=['lon','lat'])


FN='binned_drifter.npz'
Z=np.load(FN) 
xb=Z['xb']
yb=Z['yb']
ub_mean=Z['ub_mean']
ub_median=Z['ub_median']
ub_std=Z['ub_std']
ub_num=Z['ub_num']
vb_mean=Z['vb_mean']
vb_median=Z['vb_median']
vb_std=Z['vb_std']
vb_num=Z['vb_num']
Z.close()

xxb,yyb = np.meshgrid(xb, yb)
fig,axes=plt.subplots(2,2,figsize=(12,10))
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.subplots_adjust(wspace=0.07,hspace=0.07)
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[0,0].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[0,0].quiverkey(Q,0.45,0.05,0.5, r'$0.5m/s$', fontproperties={'weight': 'bold'})
axes[0,0].set_title('a')

m = Basemap(projection='cyl',llcrnrlat=41.5,urcrnrlat=42.23,\
            llcrnrlon=-70.75,urcrnrlon=-69.8,resolution='h')#,fix_aspect=False)
m.drawcoastlines(color='black')
m.ax=axes[0,0]
m.fillcontinents(color='grey',alpha=1,zorder=2)
m.drawmapboundary()
parallels = np.arange(41.5,42.23,0.1)
m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.75,-69.8,2)
m.drawmeridians(meridians,labels=[0,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
axes[0,0].set_xticklabels([])
###################################################################################33
FN='binned_model.npz'
Z1=np.load(FN) 
xb1=Z1['xb']
yb1=Z1['yb']
ub_mean1=Z1['ub_mean']
ub_median1=Z1['ub_median']
ub_std1=Z1['ub_std']
ub_num1=Z1['ub_num']
vb_mean1=Z1['vb_mean']
vb_median1=Z1['vb_median']
vb_std1=Z1['vb_std']
vb_num1=Z1['vb_num']
Z1.close()

xxb,yyb = np.meshgrid(xb, yb)
ub1 = np.ma.array(ub_mean1, mask=np.isnan(ub_mean1))
vb1 = np.ma.array(vb_mean1, mask=np.isnan(vb_mean1))
Q=axes[0,1].quiver(xxb,yyb,ub1.T,vb1.T,scale=5.)
UB1T=ub1.T
VB1T=vb1.T
qk=axes[0,1].quiverkey(Q,0.45,0.05,0.5, r'$0.5m/s$', fontproperties={'weight': 'bold'})
axes[1,0].set_title('c')
axes[0,1].set_title('b')
m = Basemap(projection='cyl',llcrnrlat=41.5,urcrnrlat=42.23,\
            llcrnrlon=-70.75,urcrnrlon=-69.8,resolution='h')#,fix_aspect=False)
m.drawcoastlines(color='black')
m.ax=axes[0,1]
m.fillcontinents(color='grey',alpha=1,zorder=2)
m.drawmapboundary()
parallels = np.arange(41.5,42.23,2)
m.drawparallels(parallels,labels=[0,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.75,-69.8,2)
m.drawmeridians(meridians,labels=[0,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)

m = Basemap(projection='cyl',llcrnrlat=41.5,urcrnrlat=42.23,\
            llcrnrlon=-70.75,urcrnrlon=-69.8,resolution='h')#,fix_aspect=False)
m.drawcoastlines(color='black')
m.ax=axes[1,0]
m.fillcontinents(color='grey',alpha=1,zorder=2)
m.drawmapboundary()
parallels = np.arange(41.5,42.23,0.1)
m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.75,-69.8,0.2)
m.drawmeridians(meridians,labels=[0,0,0,1],dashes=[1,1000],fontsize=10,zorder=0)
axes[0,1].set_yticklabels([])
axes[0,1].set_xticklabels([])
###########################################
FN='binned.npz'
#FN='binned_model.npz'
Z2=np.load(FN) 
xb2=Z2['xb']
yb2=Z2['yb']
ub_mean2=Z2['ub_mean']
ub_median2=Z2['ub_median']
ub_std2=Z2['ub_std']
ub_num2=Z2['ub_num']
vb_mean2=Z2['vb_mean']
vb_median2=Z2['vb_median']
vb_std2=Z2['vb_std']
vb_num2=Z2['vb_num']
Z2.close()

xbz=[]
ybz=[]
ub_meanz=[]
vb_meanz=[]
xxb,yyb = np.meshgrid(xb, yb)
for a in np.arange(len(xxb[0])):
    p1=[]
    p2=[]
    for b in np.arange(len(xxb)):
        
        
        if ub_num[a][b]!=0 and ub_num2[a][b]!=0:
            p1.append((ub_mean[a][b]+ub_mean2[a][b])/2.0)
            p2.append((vb_mean[a][b]+vb_mean2[a][b])/2.0)
        
        if ub_num[a][b]!=0 and ub_num2[a][b]==0:
            p1.append(ub_mean[a][b])
            p2.append(vb_mean[a][b])
        
        if ub_num[a][b]==0 and ub_num2[a][b]!=0:
            p1.append(ub_mean2[a][b])
            p2.append(ub_mean2[a][b])
            
        if ub_num[a][b]==0 and ub_num2[a][b]==0:
            p1.append((ub_mean[a][b]+ub_mean2[a][b])/2.0)
            p2.append((vb_mean[a][b]+vb_mean2[a][b])/2.0)
            
    ub_meanz.append(p1)
    vb_meanz.append(p2)
            
cc=np.arange(-1.5,1.500001,0.03)
ubz = np.ma.array(ub_meanz, mask=np.isnan(ub_meanz))
vbz = np.ma.array(vb_meanz, mask=np.isnan(vb_meanz))
Q=axes[1,1].quiver(xxb,yyb,ubz.T,vbz.T,scale=5.)
qk=axes[1,1].quiverkey(Q,0.45,0.05,0.5, r'$0.5m/s$', fontproperties={'weight': 'bold'})
data = np.genfromtxt('sea.csv',dtype=None,names=['x','y','h'],delimiter=',')    
x=[]
y=[]
h=[]
x=data['x']
y=data['y']
h=data['h']
xi = np.arange(-70.75,-69.8,0.03)
yi = np.arange(41.5,42.23,0.03)
xb,yb,hb_mean,hb_median,hb_std,hb_num = sh_bindata(x, y, h, xi, yi)
xxxb,yyyb = np.meshgrid(xb, yb)
for a in np.arange(len(hb_mean)):
    for b in np.arange(len(hb_mean[0])):
        if hb_mean[a][b]>100000:
            hb_mean[a][b]=0
           
CS=axes[1,1].contour(xxxb, yyyb, -abs(hb_mean.T),colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'),levels=[-100,-60,-30,-20,-10])
axes[1,1].clabel(CS, inline=1, fontsize=12,fmt='%4.0f')

m = Basemap(projection='cyl',llcrnrlat=41.5,urcrnrlat=42.23,\
            llcrnrlon=-70.75,urcrnrlon=-69.8,resolution='h')
m.drawcoastlines(color='black')
m.ax=axes[1,1]
m.fillcontinents(color='grey',alpha=1,zorder=2)
m.drawmapboundary()
parallels = np.arange(41.5,42.23,2)
m.drawparallels(parallels,labels=[0,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.75,-69.8,0.2)
m.drawmeridians(meridians,labels=[0,0,0,1],dashes=[1,1000],fontsize=10,zorder=0)
axes[1,1].set_title('d')
ub_nums=ub_num+ub_num2

xn=[]
yn=[]
nn=[]
for a in np.arange(len(np.hstack(xxb))):
    if  np.hstack(xxb)[a]>-70.75  and np.hstack(xxb)[a]<-69.8 and np.hstack(yyb)[a]>41.5 and np.hstack(yyb)[a]<42.23:
        xn.append(np.hstack(xxb)[a])
        yn.append(np.hstack(yyb)[a])
        nn.append(np.hstack(ub_nums.T)[a])
        
newfunc = interpolate.interp2d(xn, yn, nn, kind='cubic')
xnew = np.linspace(-70.75,-69.8,40)
ynew = np.linspace(41.5,42.23,40) 
fnew = newfunc(xnew, ynew)
xnew, ynew = np.meshgrid(xnew, ynew) 
axes[1,0].contourf(xnew, ynew, fnew, 5, cmap=plt.cm.coolwarm)
C = axes[1,0].contour(xnew, ynew, fnew, 5)
axes[1,0].clabel(C, inline=True, fontsize=12,fmt='%4.0f')
plt.savefig('Fig6.tiff',format='tiff',dpi=300,bbox_inches='tight')
plt.savefig('Fig6',dpi=300,bbox_inches='tight')
