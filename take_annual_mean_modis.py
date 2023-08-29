#Take downloaded modis data from download_modis_[var].py and get monthly& annual averages

import pandas as pd
import numpy as np
import os
from jug import TaskGenerator,barrier

@TaskGenerator
def take_averages(yr, mo, name):
    if name == 'annual':
        paths = [f"{savePath}/{str(m).zfill(2)}/{str(yr)}.pkl" for m in range(1,13)]
    else:
        paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) 
             if (f"{str(yr)}_{str(mo).zfill(2)}" in p)]

    print(name, yr, mo, paths)

    if len(paths)==0:
        return()

    i = 0
    for p in paths:
        df = pd.read_pickle(p)
        if len(vars)>1:
            if name == 'annual':
                for var in vars:
                    df[f"{var}_mean_count"] = df[var] * df[f'{var}_count'] #averages weighted by count
            else:
                for var in vars:
                    df[f"{var}_count"] = df[var]['count']
                    df[f"{var}_mean_count"] = df[var]['mean'] * df[var]['count']

                df = df.drop(vars,axis=1)
        else:
            var = vars[0]
            if name == 'annual':
                df[f"{var}_mean_count"] = df[var] * df[f'{var}_count']
            else:
                df[f"{var}_count"] = df['count']
                df[f"{var}_mean_count"] = df['mean'] * df['count']


        if i == 0:
            alldf = df.copy()
            i+=1
        else:
            alldf = alldf.add(df, fill_value=0)

    for var in vars:
        alldf[var] = alldf[f'{var}_mean_count']/alldf[f'{var}_count']

    alldf = alldf[np.concatenate([vars,[f"{var}_count" for var in vars]])]
    alldf.to_pickle(f"{savePath}/{name}/{str(yr)}.pkl")

for sat in ['aqua','terra']:
    for var in ['wv','cf','aod']:
        for time in ['day','night']:
            n = f"{sat}_{var}_{time}"

            dataPath = f"/moonbow/gleung/satlcc/MODIS_data_{n}/"
            savePath =f"/moonbow/gleung/satlcc/MODIS_{n}/"

            if not os.path.isdir(savePath):
                os.mkdir(savePath)
                
            if 'terra' in n:
                yrs = range(2001,2021)
            else:
                yrs = range(2003,2021)

            if 'cf' in n:
                vars = ['cf','cod','cth']
            elif 'wv' in n:
                vars = ['pwat']
            elif 'aod' in n:
                vars = ['aod']

            #first calculate monthly averages
            for yr in yrs:
                for mo in range(1,13):
                if not os.path.isdir(f"{savePath}{str(mo).zfill(2)}"):
                    os.mkdir(f"{savePath}/{str(mo).zfill(2)}")
                if not os.path.exists(f"{savePath}/{str(mo).zfill(2)}/{str(yr)}.pkl"):
                    take_averages(yr,mo,str(mo).zfill(2)) 

            #once all monthly averages are done
            barrier()

            #calculate annual average
            if not os.path.isdir(f"{savePath}/annual/"):
                os.mkdir(f"{savePath}/annual/")
                
            for yr in yrs:
                if not os.path.exists(f"{savePath}/annual/{str(yr)}.pkl"):
                    take_averages(yr,1,'annual')
