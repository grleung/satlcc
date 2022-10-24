import pandas as pd
import numpy as np
import os
import time
from jug import TaskGenerator,barrier

n='aqua_aod'
anaPath = "/moonbow/gleung/satlcc/MODIS_aod_data/"
savePath =f"/moonbow/gleung/satlcc/MODIS_{n}/"

if not os.path.isdir(savePath):
    os.mkdir(savePath)
print(savePath)
@TaskGenerator
def take_averages(yr, mo, name):
    if name == 'annual':
        paths = [f"{savePath}{str(m).zfill(2)}/20{str(yr).zfill(2)}.pkl" for m in range(1,13)]
    else:
        paths = [f"{anaPath}{p}" for p in sorted(os.listdir(anaPath)) 
             if (f"20{str(yr).zfill(2)}_{str(mo).zfill(2)}" in p)]
    i = 0

    for p in paths:
        df = pd.read_pickle(p)

        df['mean_count'] = df['mean'] * df['count']

        df = df[['mean_count','count']]

        if i == 0:
            alldf = df.copy()
            i+=1
        else:
            alldf = alldf.add(df, fill_value=0)
    alldf['mean'] = alldf['mean_count']/alldf['count']

    alldf.to_pickle(f"{savePath}/{name}/20{str(yr).zfill(2)}.pkl")
    
for yr in range(19,21):
    for mo in range(1,13):
       if not os.path.exists(f"{savePath}/{str(mo).zfill(2)}/20{str(yr).zfill(2)}.pkl"):
          take_averages(yr,mo,str(mo).zfill(2)) 

barrier()

for yr in range(19,21):
    if not os.path.exists(f"{savePath}/annual/20{str(yr).zfill(2)}.pkl"):
        take_averages(yr,1,'annual')
