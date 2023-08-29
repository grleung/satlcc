from retry import retry
from pydap.client import open_url
from pyresample.geometry import GridDefinition, SwathDefinition
from pyresample.kd_tree import resample_gauss
from jug import TaskGenerator
import pandas as pd
import numpy as np
import os
import xarray as xr

def sub_end(d, n=3):
    return(d[-n:])

vec_sub_end = np.vectorize(sub_end)
vec_binary_repr = np.vectorize(np.binary_repr)

#offset factor taken from metadata
sf = 0.0010000000474974513

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
        
        
        d = dataset.Water_Vapor_Near_Infrared.data[:,:]
        qc = dataset.Quality_Assurance_Near_Infrared.data[:,:]
        cld = dataset.Cloud_Mask_QA.data[:,:]

        #data is converted into a binary string of 8 bits
        qc = vec_binary_repr(qc, width=8)[:,:,0]
        #get last three digits in binary string
        qc = vec_sub_end(qc,n=4)

        #data is converted into a binary string of 8 bits
        cld = vec_binary_repr(cld, width=8)[:,:]
        #get last three digits in binary string
        cld = vec_sub_end(cld)

        out = np.where((((qc=='0101') | (qc=='0111')) & (d>=0) & ((cld=='111') | (cld=='101'))),
                        d*sf,np.nan)
        #correct values last three qc digits will be
        #0101 (good confidence) or 0111 (very good confidence)
        #correct values last three cld digits will be
        #111 (confident clear) or 101 (probably clear)
        
        swathDef = SwathDefinition(lons = longitude, lats = latitude)

        out = resample_gauss(swathDef, 
                               out,
                               gridDef, 
                               radius_of_influence=2000, 
                               sigmas = 1000,
                               fill_value=np.nan)

        out = pd.DataFrame(out,index=lats,columns=lons).stack().dropna().to_frame(name='pwat')
        out['time'] = time
        data.append(out)
        
    data = pd.concat(data)
    data = data.reset_index()
    data.columns = ['lat','lon','pwat','time']

    out = data.groupby(['lat','lon']).pwat.agg(['mean','count','std'])
    out.to_pickle(f"{anaPath}/{saveFile}")
    

for sat in ['terra','aqua']:
    for time in ['day','night']:
        dataPath = f'/moonbow/gleung/satlcc/MODIS/{sat}_wv/'
        anaPath = f'/moonbow/gleung/satlcc/MODIS_data_{sat}_wv_{time}/'

        if not os.path.isdir(anaPath):
            os.mkdir(anaPath)

        if sat=='terra':
            yrs = range(2001,2021)
        else:
            yrs = range(2003,2021)

        for yr in yrs:
            urllist = pd.read_csv(f"{dataPath}file_list_{yr}.txt",header=None)
            urllist.columns = ['url','time']
            urllist['time'] = pd.to_datetime(urllist.time)

            #separate day and night
            #terra day is 0-6 UTC, night is 12-18UTC (~10:30LT)
            #aqua day is 3-9 UTC, night is 15-21UTC (~1:30LT)
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
