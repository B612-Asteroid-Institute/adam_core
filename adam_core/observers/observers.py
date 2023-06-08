from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from mpc_obscodes import mpc_obscodes
from quivr import Float64Column, StringColumn, Table


class ObservatoryGeodetics(Table):
    code = StringColumn()
    longitude = Float64Column()
    cos_phi = Float64Column()
    sin_phi = Float64Column()
    name = StringColumn()

    def values(self):
        return np.array(self.table.select("longitude", "cos_phi", "sin_phi")).T


# Read MPC extended observatory codes file
OBSCODES = pd.read_json(mpc_obscodes, orient="index")
OBSCODES.reset_index(inplace=True, names=["code"])
OBSERVATORY_GEODETICS = ObservatoryGeodetics.from_kwargs(
    code=OBSCODES["code"].values,
    longitude=OBSCODES["Longitude"].values,
    cos_phi=OBSCODES["cos"].values,
    sin_phi=OBSCODES["sin"].values,
    name=OBSCODES["Name"].values,
)

codes: List[str] = [
    code for code in OBSERVATORY_GEODETICS.code.to_numpy(zero_copy_only=False)
]
ObservatoryCodes = Enum("ObservatoryCodes", codes)  # type: ignore