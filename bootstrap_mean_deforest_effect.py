import pandas as pd
import numpy as np
from jug import TaskGenerator
import os

binvar = 'Loss_1km'
res = 0.1
nsample = 100
nrepeat = 5000

dataPath = f"/moonbow/gleung/satlcc/deforest_effect/"
savePath = f"/moonbow/gleung/satlcc/deforest_effect/bootstrapped/"

if not os.path.isdir(savePath):
    os.mkdir(savePath)

@TaskGenerator
def bootstrap_mean(sub, var, nsample=100, nrepeat=5000):
    if len(sub)<nsample:
        return([np.nan, np.nan, np.nan])
    else:
        mns = []

        for n in range(nrepeat):
            mns.append(sub.sample(nsample)[var].mean())

        return(np.array([np.mean(mns), np.percentile(mns,25), np.percentile(mns,75)]))

@TaskGenerator
def join(partials, idx, savePath):
    out = list([*partials])
    out = pd.DataFrame(out, index = idx, columns = ['mean','ci25','ci75'])
    out.to_csv(savePath)
'''
for name in ['terra_day','terra_night','aqua_day','aqua_night']:
    for var in ['cf','cth']:
        alldf = pd.read_parquet(f"{dataPath}{name}.pq",
                                columns=[binvar, f'did{var}'])
        alldf[f"{binvar}_"] = res*(alldf[binvar]//res)
        idx = []
        subs = []

        for i, sub in alldf.groupby(f"{binvar}_"):
            idx.append(i)
            subs.append(sub)

        join([bootstrap_mean(sub,f"did{var}",nsample,nrepeat) for sub in subs], 
            idx,
            f"{savePath}{name}_{var}_{binvar}.csv")'''

for name in ['aqua_day','terra_day']:
    for var in ['cf','cth']:
        for subname in ['aod_low','aod_high','pwat_low','pwat_high']:
            for dist in ['1km','5km','10km']:
                svar = f"{subname.split('_')[0]}_{dist}"
                alldf = pd.read_parquet(f"{dataPath}{name}.pq",
                                        columns=[svar, binvar, f'did{var}'])
                
                if 'low' in subname:
                    alldf = alldf[alldf[svar]<=alldf[svar].quantile(0.25)]
                else:
                    alldf = alldf[alldf[svar]>alldf[svar].quantile(0.75)]

                alldf[f"{binvar}_"] = res*(alldf[binvar]//res)
                idx = []
                subs = []

                for i, sub in alldf.groupby(f"{binvar}_"):
                    idx.append(i)
                    subs.append(sub)

                join([bootstrap_mean(sub,f"did{var}",nsample,nrepeat) for sub in subs], 
                    idx,
                    f"{savePath}{name}_{var}_{binvar}_{subname}_{dist}_deforestyr.csv")
