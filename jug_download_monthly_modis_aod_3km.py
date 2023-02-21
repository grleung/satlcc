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

name = 'terra_aod'
dataPath = f'/moonbow/gleung/satlcc/MODIS/{name}/'
anaPath = '/moonbow/gleung/satlcc/MODIS_data_terra_aod_day/'

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
sf = 0.0010000000474974513

@TaskGenerator
@retry(delay=1, tries=20, backoff=1.1)
def download_reproj_data(urls, saveFile):
    data = []
    
    lats = np.linspace(-10, 30, 5*250*4+1)[1:][::3]
    lons = np.linspace(90, 140, 5*250*5+1)[:-1][::3]
    lons_g,lats_g = np.meshgrid(lons, lats)
    gridDef = GridDefinition(lons = lons_g, 
                                 lats = lats_g)
    
    for url,time in zip(urls.url,urls.time):
        dataset = open_url(url)

        latitude = dataset['Latitude'][:,:].data
        longitude = dataset['Longitude'][:,:].data 
        aod = dataset['Optical_Depth_Land_And_Ocean'][:,:,0].data
        aod = np.where(aod!=-9999,aod*sf,np.nan)
        
        swathDef = SwathDefinition(lons = longitude, lats = latitude)

        out = resample_gauss(swathDef, 
                               aod,
                               gridDef, 
                               radius_of_influence=2000, 
                               sigmas = 1000,
                               fill_value=np.nan)

        out = pd.DataFrame(out,index=lats,columns=lons).stack().dropna().to_frame(name='aod')
        out['time'] = time
        data.append(out)
        
    data = pd.concat(data)
    data = data.reset_index()
    data.columns = ['lat','lon','aod','time']

    out = data.groupby(['lat','lon']).aod.agg(['mean','count','std'])
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
        for i in [0,1,2,3,4,5,6,7,8,9,10]:
            urls = urllist[(urllist.time.dt.month==month) & (urllist.time.dt.day>(i*3)) & (urllist.time.dt.day<=((i+1)*3))]
            if not os.path.exists(f"{anaPath}/modis_{yr}_{str(month).zfill(2)}_{str(i).zfill(2)}.pkl"):
                if len(urls)>0:
                    #print(month,i,len(urls))
                    download_reproj_data(urls,
                                  f"modis_{yr}_{str(month).zfill(2)}_{str(i).zfill(2)}.pkl")
