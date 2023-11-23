
# Imports
import numpy as np
import xarray as xr
import dfmproc as dfmp
from pySVA.sva_helpers import reconstruct_vector_form, calculate_unit_normal_vectors

# import metpy.calc as mpcalc

def compute_velocity_perturbation(constructorSVA):
    # > First check if there's a velocity vector defined in the direction of the normal vector. This is the preferred
    # > option, because it involves less data.
    if constructorSVA.velu1 is None:
        if constructorSVA.velx or constructorSVA.vely is None:
            raise ValueError('Please make sure that the x and y velocities are defined, or give a velocity magnitude '
                             'in the direction of the normal vector .')
    
    else:
        # > Compute the mean value of the magnitude in the direction of the normal vector on the edges of the
        # > unstructured grid cell.
        if constructorSVA.velu is None:
            # > Compute the mean value of the x-velocity
            # constructorSVA.ds['mean_velx'] = \
                mean_velx = constructorSVA._mean_velx = constructorSVA.ds[f'{constructorSVA.velx.name}'].mean(dim=("depth"))

            # Calculate the vertical perturbation of the x-velocity
            # constructorSVA.ds['velx_perturbation'] = \
                velx_perturbation = constructorSVA._velx_perturbation = constructorSVA.ds[f'{constructorSVA.velx.name}'] - mean_velx

            # Compute the mean value of the y-velocity
            # constructorSVA.ds['mean_vely'] = \
                mean_vely = constructorSVA._mean_vely = constructorSVA.ds[f'{constructorSVA.vely.name}'].mean(dim=("depth"))

            # Calculate the vertical perturbation of the y-velocity
            # constructorSVA.ds['vely_perturbation'] = \
                vely_perturbation = constructorSVA._vely_perturbation = constructorSVA.ds[f'{constructorSVA.vely.name}'] - mean_vely

                return velx_perturbation, vely_perturbation

        else:
            # > Compute the mean value of the u1-velocity
            # constructorSVA.ds['mean_velu1'] = \
                mean_velu1 = constructorSVA._mean_velu1 = constructorSVA.ds[f'{constructorSVA.velu1.name}'].mean(dim=("depth"))
                velu1_perturbation = constructorSVA._velu1_perturbation = constructorSVA.ds[f'{constructorSVA.velu1.name}'] - mean_velu1

                return velu1_perturbation


def compute_tracer_variance(constructorSVA):
    if constructorSVA.tracer is None:
        raise ValueError('Please define a variable to calculate the variance of.')

    else:

        # compute mean value of tracer over depth
        constructorSVA.ds['mean_tracer'] = mean_tracer = constructorSVA._mean_tracer = constructorSVA.ds[f'{constructorSVA.tracer.name}'].mean(dim=("depth"))

        # tracer vertical perturbation
        constructorSVA.ds['tracer_perturbation'] = constructorSVA._tracer_perturbation = constructorSVA.ds[f'{constructorSVA.tracer.name}'] - mean_tracer

        # tracer variance (S'v) = (S - S_mean)
        constructorSVA.ds['tracer_variance'] = constructorSVA.tracer_variance = tracer_variance = (constructorSVA.ds[f'{constructorSVA.tracer.name}'] - mean_tracer) ** 2

        return tracer_variance

def integrate(constructorSVA):
    #todo
    return
# def compute_time_mean_variance(constructorSVA, averaging_interval):
#     """

#     :param constructorSVA:
#     :param averaging_interval: in (str, str) format #todo check
#     :return:
#     """
#     if isinstance(averaging_interval, tuple):
#         interval_start = averaging_interval[0]
#         interval_end = averaging_interval[1]
#     elif isinstance(averaging_interval, list) and len(averaging_interval) == 2:
#         interval_start = averaging_interval[0]
#         interval_end = averaging_interval[1]
#     else:
#         raise IOError('Please provide an averaging interval list or tuple format.')

#     if not 'tracer_variance' in constructorSVA.ds:
#         compute_tracer_variance(constructorSVA)
#     else:
#         pass

#     depth_mean = constructorSVA.ds[f'tracer_variance'].mean('depth') # calculate depth mean
#     time_mean = depth_mean.sel(
#         time=slice(interval_start, interval_end)).mean('time') # calculate time mean

#     # constructorSVA.ds.where((file_dataset.time.dt.hour % 4 == 1) & (file_dataset.time.dt.minute <= 20), drop=True)
#     return time_mean

def compute_advection(constructorSVA, grad_func=dfmp.compute_gradient_node_based):

    # >> 1. Calculate the salinity variance 
    if constructorSVA.tracer_variance is None:
        if constructorSVA.tracer is None:
            raise ValueError('Please define a variable to calculate the variance of.')
        else:
            compute_tracer_variance(constructorSVA)
    else:
        # >> 2. First see if there's a velocity component defined in the direction of the normal vector
        if constructorSVA.velu1 is None:
            if constructorSVA.velx is None:
                if constructorSVA.vely is None:
                    raise ValueError('Please define a horizontal velocity in x- and y-direction by either defining '
                                     'constructorSVA.velx and constructorSVA.vely or constructorSVA.velu1.')
                else:
                    raise ValueError('Please define a horizontal velocity in x-direction by defining '
                                     'constructorSVA.velx.')
            elif constructorSVA.vely is None:
                raise ValueError('Please define a horizontal velocity in y-direction by defining constructorSVA.vely.')
            # >> 3. Put the velocity vector in vector notation
            # > Define the velocities
            uhx = constructorSVA.velx
            uhy = constructorSVA.vely

            uh = reconstruct_vector_form(constructorSVA, [uhx, uhy], vector_name=f'{constructorSVA.ds.grid.name}_uc')

        else:
            unvs = calculate_unit_normal_vectors()
            uh = constructorSVA.velu1

            # >> 3. Calculate the dot product of the velocity vector times the salinity variance (scalar)
            sv2 = constructorSVA.tracer_variance
            
            # todo
            uh.dot()

            
            # uhx_sv2 = uhx * sv2
            # uhy_sv2 = uhy * sv2
            # uh_mag_sv2 = uh_mag * sv2

            # integral = uh_mag_sv2.integrate('depth')
            # grad = mpcalc.gradient(integral)  # x and y gradient of the velocity magnitude times sv2

            # x_int = uhx_sv2.mean('depth')
            # y_int = uhy_sv2.mean('depth')

            # Calculate the gradient
            # gradx = mpcalc.gradient(x_int) # x and y gradient of the velocity in x-direction times sv2
            # grady = mpcalc.gradient(y_int) # x and y gradient of the velocity in y-direction times sv2

            # grad_mag = np.sqrt(grad[0]**2+grad[1]**2)

            return grad_mag

def compute_straining(constructorSVA):
    if constructorSVA.ds.tracer_variance is None:
        if constructorSVA.tracer is None:
            raise ValueError('Please define a variable to calculate the variance of.')
        else:
            compute_tracer_variance(constructorSVA)
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
