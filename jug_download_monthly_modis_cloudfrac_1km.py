from retry import retry
from pydap.client import open_url
from pydap.cas.urs import setup_session
import time
from pyresample import kd_tree,geometry
from pyresample.plot import area_def2basemap
from pyresample import load_area, save_quicklook 
from pyresample.geometry import GridDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest, resample_gauss
from jug import TaskGenerator
import pandas as pd
import numpy as np
import os
import xarray as xr

name = 'terra_cf'
dataPath = f'/moonbow/gleung/satlcc/MODIS/{name}/'
anaPath = '/moonbow/gleung/satlcc/MODIS_data_terra_night/'

if not os.path.isdir(anaPath):
    os.mkdir(anaPath)

if 'terra' in name:
    yrs = range(2001,2021)
else:
    yrs = range(2003,2021)

def sub_end(d, n=3):
    return(d[-n:])

vec_sub_end = np.vectorize(sub_end)
vec_binary_repr = np.vectorize(np.binary_repr)

@TaskGenerator
@retry(delay=1, tries=20, backoff=1.1)
def download_reproj_data(urls, saveFile):
    data = []
    
    lats = np.linspace(-10, 30, 5*250*4+1)[1:]
    lons = np.linspace(90, 140, 5*250*5+1)[:-1]
    lons_g,lats_g = np.meshgrid(lons, lats)
    gridDef = GridDefinition(lons = lons_g, 
                                 lats = lats_g)
    
    for url,time in zip(urls.url,urls.time):
        dataset = open_url(url)#url.split('RemoteResources/laads/')[0]+
                           #url.split('RemoteResources/laads/')[1])

        latitude = dataset['Latitude_1'][:,:].data
        longitude = dataset['Longitude_1'][:,:].data #Longitude_1 for 1km
        cm = dataset['Cloud_Mask_1km'][:,:,0].data
        cm = vec_binary_repr(cm, width=8)[:,:,0]
        cm = vec_sub_end(cm)

        cf = np.ones(cm.shape)
        cf = np.where(((cm=='001') | (cm=='011')),cf,0)

        sf = 0.009999999776482582
        cot = dataset.Cloud_Optical_Thickness.data[:,:]
        cot = np.where(((cm=='001') | (cm=='011')) & (cot!=-9999),cot*sf,np.nan)

        cth = dataset.cloud_top_height_1km.data[:,:]
        cth = np.where(((cm=='001') | (cm=='011')) & (cth>0),cth,np.nan)

        df = np.dstack((cf,cot,cth))
        
        swathDef = SwathDefinition(lons = longitude, lats = latitude)

        out = resample_gauss(swathDef, 
                               df,
                               gridDef, 
                               radius_of_influence=2000, 
                               sigmas = [1000,1000,1000],
                               fill_value=np.nan)

        out = xr.Dataset(data_vars={'cf':(['lat','lon'],out[:,:,0]),
                            'cod':(['lat','lon'],out[:,:,1]),
                            'cth':(['lat','lon'],out[:,:,2])},
                coords = {'lat':('lat',lats),
                            'lon':('lon',lons)}).to_dataframe().dropna(how='all')

        out['time'] = time
        data.append(out)
        
    data = pd.concat(data)
    data = data.reset_index()
    data.columns = ['lat','lon','cf','cod','cth','time']

    out = data.groupby(['lat','lon'])[['cf','cod','cth']].agg(['mean','count','std'])
    out.to_pickle(f"{anaPath}/{saveFile}")
    

for yr in yrs:
    urllist = pd.read_csv(f"{dataPath}file_list_{yr}.txt",header=None)
    urllist.columns = ['url','time']
    urllist['time'] = pd.to_datetime(urllist.time)

    if 'day' in anaPath:
        urllist = urllist[urllist.time.dt.hour<12]
    else:
        urllist = urllist[urllist.time.dt.hour>=12]

    for month in range(1,13):
        for i in np.arange(32):
            urls = urllist[(urllist.time.dt.month==month) & (urllist.time.dt.day==i)]
            if not os.path.exists(f"{anaPath}/modis_{yr}_{str(month).zfill(2)}_{str(i).zfill(2)}.pkl"):
                if len(urls)>0:
                    download_reproj_data(urls,
                                  f"modis_{yr}_{str(month).zfill(2)}_{str(i).zfill(2)}.pkl")
