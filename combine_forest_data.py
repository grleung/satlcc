# This script takes GFC and GSW data and merges it into one file of the right (~1km) resolution

import os
import numpy as np
import pandas as pd

alldata = []

# specified region
for lat in [0, 10, 20, 30]:
    for lon in [90, 100, 110, 120, 130]:
        print(lat, lon)
        lats = [round(l, 3) for l in np.linspace(lat - 10, lat, 1250 + 1)][1:]
        lons = [round(l, 3) for l in np.linspace(lon, lon + 10, 1250 + 1)][:-1]

        # read GFC data
        forest = np.load(
            f"{gfcPath}forestcover_{str(lat).zfill(2)}N_{str(lon).zfill(3)}E.npy"
        )

        data = []
        for yr in range(21):
            out = pd.DataFrame(
                forest[:, :, yr], index=lats, columns=lons
            ).stack()
            out.name = f"Forest_{yr}"

            data.append(out)

        data = pd.concat(data, axis=1)

        # read GSW data
        water = pd.DataFrame(
            np.load(
                f"{gswPath}watercover1km_{str(lat).zfill(2)}N_{str(lon).zfill(3)}E.npy"
            ),
            index=lats,
            columns=lons,
        ).stack()

        data["Water"] = water

        # include only pixels where surface water coverage is below 1%
        data = data[data.Water < 0.01]

        alldata.append(data)

alldata = pd.concat(alldata, axis=0)

# calc loss
for yr in range(1, 21):
    alldata[f"Loss_{yr}"] = -(
        alldata[f"Forest_{yr}"] - alldata[f"Forest_{yr-1}"]
    )

# save as parquet
alldata.to_parquet(path=f"{anaPath}/base_forest_data.pq", engine="pyarrow")
