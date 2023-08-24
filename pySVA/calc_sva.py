
# Imports
import numpy as np
import xarray as xr
# import metpy.calc as mpcalc

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

    else:

        # compute mean value of tracer over depth
        constructorSVA.ds['mean_tracer'] = mean_tracer = constructorSVA._mean_tracer = constructorSVA.ds[f'{constructorSVA.tracer.name}'].mean(dim=("depth"))

        # tracer vertical perturbation
        constructorSVA.ds['tracer_perturbation'] = constructorSVA._tracer_perturbation = constructorSVA.ds[f'{constructorSVA.tracer.name}'] - mean_tracer

        # tracer variance (S'v)^2 = (S - S_mean)^(2)
        constructorSVA.ds['tracer_variance'] = constructorSVA.tracer_variance = tracer_variance = (constructorSVA.ds[f'{constructorSVA.tracer.name}'] - mean_tracer) ** 2

        return tracer_variance

def compute_time_mean_variance(constructorSVA, averaging_interval):
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

    depth_mean = constructorSVA.ds[f'tracer_variance'].mean('depth') # calculate depth mean
    time_mean = depth_mean.sel(
        time=slice(interval_start, interval_end)).mean('time') # calculate time mean

    # constructorSVA.ds.where((file_dataset.time.dt.hour % 4 == 1) & (file_dataset.time.dt.minute <= 20), drop=True)
    return time_mean

def compute_advection(constructorSVA):

    if constructorSVA.ds.tracer_variance is None:
        if constructorSVA.tracer is None:
            raise ValueError('Please define a variable to calculate the variance of.')
        else:
            compute_variance(constructorSVA)
    else:

        if constructorSVA.ds.velx is None:
            if constructorSVA.ds.vely is None:
                raise ValueError('Please define a horizontal velocity in x and y direction by defining constructorSVA.velx and constructorSVA.vely.')
            else:
                raise ValueError('Please define a horizontal velocity in x direction by defining constructorSVA.velx.')
        elif constructorSVA.ds.vely is None:
            raise ValueError('Please define a horizontal velocity in y direction by defining constructorSVA.vely.')
        else:

            # First compute the horizontal velocity vector times the vertical salinity variance sv2 (scalar)
            uhx = constructorSVA.velx
            uhy = constructorSVA.vely
            uh_mag = np.sqrt(uhx**2+uhy**2)

            sv2 = constructorSVA.ds.tracer_variance
            # uhx_sv2 = uhx * sv2
            # uhy_sv2 = uhy * sv2
            uh_mag_sv2 = uh_mag * sv2

            integral = uh_mag_sv2.integrate('depth')
            # grad = mpcalc.gradient(integral)  # x and y gradient of the velocity magnitude times sv2

            # x_int = uhx_sv2.mean('depth')
            # y_int = uhy_sv2.mean('depth')

            # Calculate the gradient
            # gradx = mpcalc.gradient(x_int) # x and y gradient of the velocity in x-direction times sv2
            # grady = mpcalc.gradient(y_int) # x and y gradient of the velocity in y-direction times sv2

            grad_mag = np.sqrt(grad[0]**2+grad[1]**2)

            return grad_mag

def compute_straining(constructorSVA):
    if constructorSVA.ds.tracer_variance is None:
        if constructorSVA.tracer is None:
            raise ValueError('Please define a variable to calculate the variance of.')
        else:
            compute_variance(constructorSVA)
    else:
        if constructorSVA.ds.velz is None:
            raise ValueError('Please define a horizontal velocity in z direction by defining constructorSVA.velz.')

    uv = constructorSVA.velz
    uv_mean = constructorSVA.velz.mean("depth")
    uv_prime = uv - uv_mean

    sv = constructorSVA._tracer_perturbation
    s_mean = constructorSVA._mean_tracer

    grad_s_mean = mpcalc.gradient(s_mean)

    straining = -2 * uv_prime * sv * grad_s_mean
    straining_int = straining.integrate('depth')

    return straining_int


def compute_dissipation(constructorSVA):
    if constructorSVA.ds.tracer_variance is None:
        if constructorSVA.tracer is None:
            raise ValueError('Please define a variable to calculate the variance of.')
        else:
            compute_variance(constructorSVA)
        if constructorSVA.kzz is None:
            raise ValueError('Please define the vertical diffusion.')

    # in D-FLOW: kzz = uda.mesh2d_vicwwu/0.7 + 5e-5 + (1/700)*1e-6
    kzz = constructorSVA.kzz
    ds_dz = constructorSVA.tracer.differentiate(coord='depth')
    dissipation = 2 * kzz * ds_dz**2

    dissipation_int = dissipation.integrate(coord='depth')

    return dissipation_int
