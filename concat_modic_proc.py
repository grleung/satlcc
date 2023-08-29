import numpy as np
import pandas as pd
import os

for name in ['terra_day','terra_night','aqua_day','aqua_night']:
    anaPath = f"/moonbow/gleung/satlcc/deforest_effect/{name}/"

    print(sorted(os.listdir(anaPath)))
    alldf = []

    if 'aqua' in name:
        yrs = np.arange(4,20)
    else:
        yrs = np.arange(2,20)
        
    for yr in yrs:
        df = pd.read_parquet(f"{anaPath}{yr}.pq")

        df['Loss'] = df[f'Loss_{yr}']

        for dist in range(1,11):
            df[f'Loss_{dist}km'] = df[f'Loss_{yr}_{dist}km']

        df = df[np.concatenate([['PriorLoss','year',],
                                [f"Loss_{dist}km" for dist in range(1,11)],
                                [f"did{var}" for var in ['cf','cth','cod']],
                                [f"{var}_{yr-1}" for var in ['cf','cth','cod']],
                                [f"{var}_{yr+1}" for var in ['cf','cth','cod']],
                                np.concatenate([[[[f"{var}_{dist}km{n}" for var in ['aod','pwat']] for dist in [1,5,10]] for n in ['','_pre','_post','_mean']]]).flatten()]
                )]

        print(yr, len(df))
        alldf.append(df)

    alldf = pd.concat(alldf)

    alldf.to_parquet(f"{anaPath}../{name}.pq")