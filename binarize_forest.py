import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from jug import TaskGenerator

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

dataPath = '/moonbow/gleung/satlcc/GFC_2021_v1.9/'
figPath = '/moonbow/gleung/satlcc-figures/presentation/'
anaPath = '/moonbow/gleung/satlcc/GFC_processing_10km/'

if not os.path.isdir(anaPath):
    os.mkdir(anaPath)

@nb.njit
def binarize_annual_forest(treecover,lossyear,forest_thresh=75,yrs = 21):
    #binarizing forest cover
    forest_00 = np.where(treecover>=forest_thresh,1,0)
    
    forest_all = np.zeros((forest_00.shape[0],forest_00.shape[1],yrs))
    
    for yr in range(forest_all.shape[2]):
        forest_all[:,:,yr] = forest_00[:,:] #initially set forest cover for all years to be the same as in year 2000
    
    for i in range(forest_all.shape[0]):
        for j in range(forest_all.shape[1]):
            if (forest_00[i,j]!=0) & (lossyear[i,j]!=0): #if not forested to begin with, then no need to look at it
                loss_yr = lossyear[i,j]
                for yr in range(loss_yr,forest_all.shape[2]): #replace all forest cover values in years after loss year with 0
                    forest_all[i,j,yr] = 0

    return(forest_all)

@nb.njit
def coarsen_forest(forest,ave_res=32,yrs=21):
    n = int(forest.shape[0]/ave_res)
    forest_coarse = np.zeros((n,n,yrs))
    
    for yr in range(yrs):
        for i in range(n):
            for j in range(n):
                forest_coarse[i,j,yr] = forest[ave_res*i:ave_res*(i+1),ave_res*j:ave_res*(j+1),yr].sum()/(ave_res*ave_res)
                
    return(forest_coarse)

@TaskGenerator
def run(lat, lon, ns, ew,ave_res,n):
    print(lat, lon)
    var = 'treecover2000'
    path = f"{dataPath}{var}/Hansen_GFC-2021-v1.9_{var}_{str(lat).zfill(2)}{ns}_{str(lon).zfill(3)}{ew}.tif"
    treecover = np.array(Image.open(path))[::-1,::]

    var = 'lossyear'
    path = f"{dataPath}{var}/Hansen_GFC-2021-v1.9_{var}_{str(lat).zfill(2)}{ns}_{str(lon).zfill(3)}{ew}.tif"
    lossyear = np.array(Image.open(path))[::-1,::]

    forest = []
    for i in range(0,int(treecover.shape[0]/(ave_res*n))):
        forest_rows = []
        for j in range(0,int(treecover.shape[1]/(ave_res*n))):
            treecover_ = treecover[i*ave_res*n:(i+1)*ave_res*n,j*ave_res*n:(j+1)*ave_res*n]
            lossyear_ = lossyear[i*ave_res*n:(i+1)*ave_res*n,j*ave_res*n:(j+1)*ave_res*n]

            forest_all=binarize_annual_forest(treecover_,lossyear_,forest_thresh)
            forest_coarse = coarsen_forest(forest_all,ave_res)

            forest_rows.append(forest_coarse)

        forest.append(np.concatenate(forest_rows,axis=1))

    print('saving')
    forest = np.concatenate(forest,axis=0)
    np.save(f"{anaPath}forestcover10km_{str(lat).zfill(2)}{ns}_{str(lon).zfill(3)}{ew}", forest)

forest_thresh=75
ave_res = 320 #32 arcseconds is approximately 1km
n = 2 #number of chunks to subset (~number of averes km x number of km box)

for lat_ in np.arange(-20,-10,10):
    for lon_ in np.arange(-180,180,10):
        if lat_<0:
            ns = 'S'
            lat = -lat_
        else:
            ns = 'N'
            lat = lat_

        if lon_<0:
            ew = 'W'
            lon = -lon_
        else:
            ew = 'E'
            lon = lon_

        if not os.path.exists(f"{anaPath}forestcover10km_{str(lat).zfill(2)}{ns}_{str(lon).zfill(3)}{ew}.npy"):
            run(lat,lon, ns, ew,ave_res,n)
