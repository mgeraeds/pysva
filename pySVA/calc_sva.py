
# Imports
import numpy as np
import xarray as xr

# tef.tracer = tef.ds.salt
# data_description = {
#     "lon" : "mesh2d_face_node_x",
#     "lat" : "mesh2d_face_node_y",
#     "depth" : "mesh2d_layer_sigma_z",
#     "time" : "time"
# }

def compute_variance(constructorSVA):
    if constructorSVA.tracer is None:
        raise ValueError('Please define a variable to calculate the variance of.')

        time_len = constructorSVA.ds.shape[0]

        # compute mean value of tracer over depth
        contructorSVA.ds['mean_tracer'] = contructorSVA.ds[f'{constructorSVA.tracer}'].mean(dim=("layers"))

        # tracer variance (S'v)^2 = (S - S_mean)^(2)
        contructorSVA.ds['tracer_variance'] = (contructorSVA.ds[f'{constructorSVA.tracer}'] - ds_int.mean_salinity) ** 2
        tracer_variance = contructorSVA.ds['tracer_variance']

        return tracer_variance

def compute_bulk(constructorSVA, averaging_interval):
    """

    :param constructorSVA:
    :param averaging_interval: in (str, str) format #todo check
    :return:
    """
    if isinstance(averaging_interval, tuple):
        interval_start = averaging_interval[0]
        interval_end = averaging_interval[1]
    elif isinstance(averaging_interval, list) and len(averaging_interval) == 2:
        interval_start = averaging_interval[0]
        interval_end = averaging_interval[1]
    else:
        raise IOError('Please provide an averaging interval list or tuple format.')

    if not 'tracer_variance' in constructorSVA.ds:
        compute_variance(constructorSVA)
    else:
        pass

    depth_mean = constructorSVA.ds[f'tracer_variance'].mean('layers') # calculate depth mean
    time_mean = depth_mean.sel(
        time=slice(interval_start, interval_end)).mean('time') # calculate time mean

    bulk_tracer_variance = time_mean

    # constructorSVA.ds.where((file_dataset.time.dt.hour % 4 == 1) & (file_dataset.time.dt.minute <= 20), drop=True)
    return bulk_tracer_variance
