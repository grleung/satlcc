# Once estimated deforestation effect (epsilon) has been calculated in calc_deforest_effect.py,
# this script will bootstrap the mean and 25th/75th percentile estimate for each forest loss bin

import pandas as pd
import numpy as np
from jug import TaskGenerator

# variable to bin over, here using mean forest loss within 1km radius
binvar = "Loss_1km"
# resolution of binvariable (this means we look at forest loss from 0-0.1, 0.1-0.2, etc.)
res = 0.1

# number of samples to draw and number of times to repeat bootstrapping
nsample = 100
nrepeat = 5000

# where to find output of calc_deforest_effect.py
dataPath = f"/moonbow/gleung/satlcc/deforest_effect/"


@TaskGenerator
def bootstrap_mean(sub, var, nsample=100, nrepeat=5000):
    # This function takes the bootstraped mean of some dataframe "sub" for some variable "var"
    # output is the mean and 25th/75th percentile

    if len(sub) < nsample:
        # if there are not enough values in the dataframe to draw nsample number of samples
        # return nan values
        return [np.nan, np.nan, np.nan]
    else:
        # if there are sufficient number of samples, do bootstrapping
        mns = []

        for n in range(nrepeat):
            mns.append(sub.sample(nsample)[var].mean())

        return np.array(
            [np.mean(mns), np.percentile(mns, 25), np.percentile(mns, 75)]
        )


@TaskGenerator
def join(partials, idx, savePath):
    # join the bootstrapped parameters for each forest loss bin into one dataframe that we save
    out = list([*partials])
    out = pd.DataFrame(out, index=idx, columns=["mean", "ci25", "ci75"])
    out.to_csv(savePath)


# Overall means - This is shown in Figure 2 of main text
for name in ["aqua_day", "terra_night", "aqua_day", "aqua_night"]:
    for var in ["cf", "cth"]:
        # read in out put of calc_deforest_effect.py
        alldf = pd.read_parquet(
            f"{dataPath}{name}.pq", columns=[binvar, f"did{var}"]
        )

        # convert binvar to specified resolution
        alldf[f"{binvar}_"] = res * (alldf[binvar] // res)

        # temporary lists to feed to jug tasks
        idx = []
        subs = []
        for i, sub in alldf.groupby(f"{binvar}_"):
            idx.append(i)
            subs.append(sub)

        # call jug task bootstrap estimator for each forest loss bin, then save in specified path
        join(
            [
                bootstrap_mean(sub, f"did{var}", nsample, nrepeat)
                for sub in subs
            ],
            idx,
            f"{dataPath}/bootstrapped/{name}_{var}_{binvar}.csv",
        )

# By Environmental Variables - This is shown in Figure 3-4 of main text
for name in ["aqua_day", "terra_day"]:
    for var in ["cf", "cth"]:
        for subname in ["aod_low", "aod_high", "pwat_low", "pwat_high"]:
            for dist in [
                "1km",
                "5km",
                "10km",
            ]:  # testing various averaging ranges for environmental variable
                svar = f"{subname.split('_')[0]}_{dist}"

                # read in output of calc_deforest_effect.py
                alldf = pd.read_parquet(
                    f"{dataPath}{name}.pq", columns=[svar, binvar, f"did{var}"]
                )

                # thresholds for low vs. high quartiles
                # these are used to subset the data according to some environmental variable "svar"
                if "low" in subname:
                    alldf = alldf[alldf[svar] <= alldf[svar].quantile(0.25)]
                else:
                    alldf = alldf[alldf[svar] > alldf[svar].quantile(0.75)]

                # convert binvar to specified resolution
                alldf[f"{binvar}_"] = res * (alldf[binvar] // res)

                # temporary lists to feed to jug tasks
                idx = []
                subs = []
                for i, sub in alldf.groupby(f"{binvar}_"):
                    idx.append(i)
                    subs.append(sub)

                # call jug task bootstrap estimator for each forest loss bin, then save in specified path
                join(
                    [
                        bootstrap_mean(sub, f"did{var}", nsample, nrepeat)
                        for sub in subs
                    ],
                    idx,
                    f"{dataPath}/bootstrapped/{name}_{var}_{binvar}_{subname}_{dist}.csv",
                )

# By year - this is sensitivity testing for Figure S2
for name in ["aqua_day", "terra_day", "terra_night", "aqua_night"]:
    for var in ["cf", "cth"]:
        # read in out put of calc_deforest_effect.py
        alldf = pd.read_parquet(
            f"{dataPath}{name}.pq",
            columns=[binvar, f"did{var}", "year"],
        )
        # convert binvar to specified resolution
        alldf[f"{binvar}_"] = res * (alldf[binvar] // res)

        # do for each year
        for yr in sorted(alldf.year.unique()):
            # temporary lists to feed to jug tasks
            idx = []
            subs = []

            s = alldf[alldf.year.isin([yr])]

            for i, sub in s.groupby(f"{binvar}_"):
                idx.append(i)
                subs.append(sub)

            # call jug task bootstrap estimator for each forest loss bin, then save in specified path
            join(
                [
                    bootstrap_mean(sub, f"did{var}", nsample, nrepeat)
                    for sub in subs
                ],
                idx,
                f"{dataPath}/bootstrapped/{name}_{var}_{binvar}_{yr}_only.csv",
            )
