import xarray as xr
import numpy as np
ds = xr.open_dataset('data/ersst.v4.195409.nc')
vals = ds['sst'].values
vals.count()
