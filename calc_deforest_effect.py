
import os
import pandas as pd
import numpy as np
from astropy.convolution import convolve
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
import scipy.stats as sts
pd.options.mode.chained_assignment = None  # default='warn'
import pyarrow
from sklearn.neighbors import KDTree
from jug import TaskGenerator

dataPath = '/moonbow/gleung/satlcc/GFC_2021_v1.9/'
figPath = '/moonbow/gleung/satlcc-figures/deforest_effect/'
gfcPath = '/moonbow/gleung/satlcc/GFC_processing/'
gswPath = '/moonbow/gleung/satlcc/GSW_processing/'
anaPath = '/moonbow/gleung/satlcc/deforest_effect/'

for path in [figPath,anaPath]:
    if not os.path.isdir(path):
        os.mkdir(path)
        
char = 'abcdefghijklmnopqrstuvwxyz'

def mean_within_radius(var,sub,rad=10,ind=('level_0','level_1')):
    data = sub.reset_index().pivot(index=ind[0],columns=ind[1],values=var)
    
    #because the grid is 1km resolution, a distance X km away is just X points
    kernel = np.fromfunction(lambda x, y: ((x-rad)**2 + (y-rad)**2 <= rad**2)*1, 
                             (2*rad+1, 2*rad+1), dtype=int).astype(np.uint8)
    kernel = kernel/np.sum(kernel)

    #astropy is able to handle nan
    out = convolve(data,kernel,
                   boundary='fill',fill_value=np.nan, 
                   nan_treatment='interpolate',
                   preserve_nan=True)

    temp = pd.DataFrame(data.stack(dropna=False))
    temp.loc[:,f"{var}_{rad}km"] = out.ravel()

    return(temp[f"{var}_{rad}km"])

def max_within_radius(var,sub,rad=10,ind=('level_0','level_1')):
    data = sub.reset_index().pivot(index=ind[0],columns=ind[1],values=var)
    
    #because the grid is 1km resolution, a distance X km away is just X points
    kernel = np.fromfunction(lambda x, y: ((x-rad)**2 + (y-rad)**2 <= rad**2)*1, 
                             (2*rad+1, 2*rad+1), dtype=int).astype(np.uint8)
     
    #substitute nan with 0
    out = maximum_filter(np.nan_to_num(data), footprint=kernel,
                    mode = 'nearest')

    temp = pd.DataFrame(data.stack(dropna=False))
    temp.loc[:,f"{var}_{rad}km_max"] = out.ravel()

    return(temp[f"{var}_{rad}km_max"])

@TaskGenerator
def run(yr,name):
    modisPath = f"/moonbow/gleung/satlcc/MODIS_{name.split('_')[0]}_cf_{name.split('_')[1]}"
    aodPath = f"/moonbow/gleung/satlcc/MODIS_{name.split('_')[0]}_aod_{name.split('_')[1]}"
    wvPath = f"/moonbow/gleung/satlcc/MODIS_{name.split('_')[0]}_wv_{name.split('_')[1]}"

    if not os.path.isdir(f"{anaPath}/{name}"):
        os.mkdir(f"{anaPath}/{name}")

    alldata = pd.read_parquet(f"{anaPath}/base_forest_data.pq",engine='pyarrow')
    
    #define all years prior to year in question 
    prior = [f"Loss_{yr}" for yr in range(1,yr+1)]

    #subset just the data needed, initial forest cover and loss all years prior to this year
    sub = alldata[[f"Forest_0"]+ prior]
    #calculate total prior loss before this year
    sub['PriorLoss'] = sub[prior].sum(axis=1)
    
    
    #read cloud fraction from years before and after this year
    cf_pre = pd.read_pickle(f"{modisPath}/annual/20{str(yr-1).zfill(2)}.pkl")
    cf_post = pd.read_pickle(f"{modisPath}/annual/20{str(yr+1).zfill(2)}.pkl")
    cf_pre.index =  [(round(lat_,3), round(lon_,3)) for lat_, lon_ in cf_pre.index]
    cf_post.index =  [(round(lat_,3), round(lon_,3)) for lat_, lon_ in cf_post.index]
    
    for var in ['cf','cth','cod']:
        sub[f"{var}_{yr-1}"] = sub.index.map(cf_pre[var])
        sub[f"{var}_{yr+1}"] = sub.index.map(cf_post[var])
        sub[f'delta{var}'] = sub[f"{var}_{yr+1}"]-sub[f"{var}_{yr-1}"]
    
    #read AOD/WV
    pwat_pre = pd.read_pickle(f"{wvPath}/annual/20{str(yr-1).zfill(2)}.pkl")
    pwat = pd.read_pickle(f"{wvPath}/annual/20{str(yr).zfill(2)}.pkl")
    pwat_post = pd.read_pickle(f"{wvPath}/annual/20{str(yr+1).zfill(2)}.pkl")

    for var in ["pwat"]:
        for dist in [1,5,10]:
            pwat[f"{var}_{dist}km"] = mean_within_radius(var,pwat,dist,ind=('lat','lon'))
            pwat[f"{var}_{dist}km_pre"] = mean_within_radius(var,pwat_pre,dist,ind=('lat','lon'))
            pwat[f"{var}_{dist}km_post"] = mean_within_radius(var,pwat_post,dist,ind=('lat','lon'))
            pwat[f"delta{var}_{dist}km"] = pwat[f"{var}_{dist}km_post"] - pwat[f"{var}_{dist}km_pre"]
            
    pwat.index =  [(round(lat_,3), round(lon_,3)) for lat_, lon_ in pwat.index]

    for var in ['pwat_1km','pwat_1km_pre','pwat_1km_post',
                'pwat_5km','pwat_5km_pre','pwat_5km_post',
                'pwat_10km','pwat_10km_pre','pwat_10km_post',
                'deltapwat_1km','deltapwat_5km','deltapwat_10km',
                ]:
        sub[f"{var}"] = sub.index.map(pwat[var])

    for dist in [1,5,10]:
        sub[f'pwat_{dist}km_mean'] = sub[[c for c in sub.columns if (f'pwat_{dist}km' in c) & ('delta' not in c)]].mean(axis=1)
        sub[f"pipwat_{dist}km"] = sub[f'deltapwat_{dist}km']/sub[f'pwat_{dist}km_mean']

    aod_pre = pd.read_pickle(f"{aodPath}/annual/20{str(yr-1).zfill(2)}.pkl")
    aod = pd.read_pickle(f"{aodPath}/annual/20{str(yr).zfill(2)}.pkl")
    aod_post = pd.read_pickle(f"{aodPath}/annual/20{str(yr+1).zfill(2)}.pkl")

    for var in ["aod"]:
        for dist in [1,5,10]:
            aod[f"{var}_{dist}km"] = mean_within_radius(var,aod,dist,ind=('lat','lon'))
            aod[f"{var}_{dist}km_pre"] = mean_within_radius(var,aod_pre,dist,ind=('lat','lon'))
            aod[f"{var}_{dist}km_post"] = mean_within_radius(var,aod_post,dist,ind=('lat','lon'))

    aod.index =  [(round(lat_,3), round(lon_,3)) for lat_, lon_ in aod.index]

    for var in ['aod_1km','aod_1km_pre','aod_1km_post',
                'aod_5km','aod_5km_pre','aod_5km_post',
                'aod_10km','aod_10km_pre','aod_10km_post']:
        sub[f"{var}"] = sub.index.map(aod[var])

    for dist in [1,5,10]:
        sub[f'aod_{dist}km_mean'] = sub[[c for c in sub.columns if f'aod_{dist}km' in c]].mean(axis=1)

    #calculating maximum and mean values within 10km radius of pixel
    for var in ['PriorLoss',f"Loss_{yr}"]:
        for dist in [1,2,3,4,5,6,7,8,9,10]:
            sub[f"{var}_{dist}km"] = mean_within_radius(var,sub,dist)
    
        sub[f"{var}_10km_max"] = max_within_radius(var,sub)
    
    #define population of possible control points: 
    #(1) high initial forest cover
    #(2) low forest loss prior to this year 
    #(3) low mean forest loss within 10km radius prior to this year
    ctrl = sub[(sub.Forest_0>=0.9) &
             (sub.PriorLoss<=0.02) &
             (sub.PriorLoss_10km<=0.02)]
    ctrlpts = np.array(list(ctrl.index.values))
    
    #define population of assessment points:
    #any pixel within 10km of a pixel which lost 50% forest cover this year
    defo = sub[sub[f"Loss_{yr}_10km_max"]>=0.5]

    #remove any assessment pixels which don't have an equivalent control pixel within 25km
    #we can't pair them with control pixel so not included in final stats
    defopts = np.array(list(defo.index.values))
    n =  KDTree(ctrlpts).query_radius(defopts, 25*0.008, count_only=True)
    defopts = defopts[n!=0]
    defo = defo.loc[list(zip(defopts[:,0],defopts[:,1]))]
    
    
    if len(defo)!=0:
        #for each remaining assessment pixel, find control points which are near enough (within 25km)
        #to serve as control for this pixel
        #take mean cloud cover
        for pos in defo.index:
            ctrl_sub = ctrlpts[(abs(ctrlpts[:,0]-pos[0]) <= 25*0.008) & (abs(ctrlpts[:,1]-pos[1]) <= 25*0.008)] #exclude points which are too far in one direction anyway
            
            n =  KDTree(ctrl_sub).query_radius(np.array(list(pos)).reshape(-1,1).T, 25*0.008)
            ctrl_sub = ctrl_sub[n[0]]
            ctrl_sub = ctrl.loc[list(zip(ctrl_sub[:,0],ctrl_sub[:,1]))]

            for var in ['cf','cth','cod',
                'pwat_1km','pwat_5km','pwat_10km']:
                defo.loc[pos,f'delta{var}_ctrl'] = ctrl_sub[f'delta{var}'].mean()
                    
        #this is episilon/difference-in-differences metric
        for var in ['cf','cth','cod','pwat_1km','pwat_5km','pwat_10km']:
            defo[f'did{var}'] = defo[f'delta{var}'] - defo[f'delta{var}_ctrl']


        #just save needed columns
        defo = defo[np.concatenate([[f'Loss_{yr}','PriorLoss',
                                    f'Loss_{yr}_10km_max',f'PriorLoss_10km_max'],
                                    [f"Loss_{yr}_{dist}km" for dist in range(1,11)],
                                    [f"PriorLoss_{dist}km" for dist in range(1,11)],
                                    [f"did{var}" for var in ['cf','cth','cod', 'pwat_1km','pwat_5km','pwat_10km']],
                                    [f"{var}_{yr-1}" for var in ['cf','cth','cod']],
                                    [f"{var}_{yr+1}" for var in ['cf','cth','cod']],
                                    ['deltapwat_1km','deltapwat_5km','deltapwat_10km'],
                                    ['pipwat_1km','pipwat_5km','pipwat_10km'],
                                    np.concatenate([[[[f"{var}_{dist}km{n}" for var in ['aod','pwat']] for dist in [1,5,10]] for n in ['','_pre','_post','_mean']]]).flatten()
            ])]

        defo['year'] = yr

        defo.to_parquet(f"{anaPath}/{name}/{yr}.pq", engine='pyarrow')


yrs = range(2,20)
# TERRA goes from 2001-2020, AQUA from 2003-2020
# DID metric can be calculated for 2002-2019 or 2004-2019, respectively

for name in ['terra_day','aqua_day']:
    for yr in yrs:
        if os.path.exists(f"/moonbow/gleung/satlcc/MODIS_{name.split('_')[0]}_cf_{name.split('_')[1]}/annual/20{str(yr-1).zfill(2)}.pkl"):
            run(yr, name)
