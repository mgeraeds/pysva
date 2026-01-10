"""Calculation module for pySVA containing functions for variance analysis.

This module provides functions to compute tracer variance, advection, straining,
and dissipation terms for salinity variance analysis.
"""

# Imports
import numpy as np
import xarray as xr
import dfmproc as dfmp
from dfm_tools.xugrid_helpers import get_vertical_dimensions
from pySVA.sva_helpers import reconstruct_vector_form, calculate_unit_normal_vectors, compute_gradient_on_face, uda_to_edges

def compute_tracer_variance(constructorSVA):
    """Compute tracer variance (S'² = (S - S_mean)²).

    Calculates the vertical mean tracer, perturbation from mean,
    and squared perturbation (variance). Results are stored in the
    constructorSVA.ds dataset.

    Args:
        constructorSVA (constructorSVA): Object containing dataset and tracer variable.

    Returns:
        xarray.DataArray: Squared tracer perturbation (variance).

    Raises:
        ValueError: If no tracer variable is defined in constructorSVA.
    """
    # > Get dataset
    uds = constructorSVA.ds
    # > Derive dimension names from dataset
    grid = uds.grid
    dimn_faces = grid.face_dimension
    dimn_edges = grid.edge_dimension
    dimn_layer, dimn_interfaces = get_vertical_dimensions(uds)

    if constructorSVA.tracer is None:
        raise ValueError('Please define a variable to calculate the variance of.')

    else:

        # > Compute mean value of tracer over depth
        constructorSVA.ds['mean_tracer'] = mean_tracer = constructorSVA._mean_tracer = constructorSVA.ds[f'{constructorSVA.tracer.name}'].mean(dim=(dimn_layer))

        # > Tracer vertical perturbation
        constructorSVA.ds['tracer_perturbation'] = constructorSVA._tracer_perturbation = constructorSVA.ds[f'{constructorSVA.tracer.name}'] - mean_tracer

        # > Tracer variance (S'v) = (S - S_mean)
        constructorSVA.ds['tracer_variance'] = constructorSVA.tracer_variance = tracer_variance = (constructorSVA.ds[f'{constructorSVA.tracer.name}'] - mean_tracer) ** 2

        return tracer_variance

def compute_advection(constructorSVA, gradient_function=None):
    """Compute the advection term (∇·(uh·S'²)).

    Calculates the flux of variance due to mean flow advection.
    Integrates over depth and applies gradient operator to obtain
    the divergence of advective flux.

    Args:
        constructorSVA (constructorSVA): Object containing dataset, velocities, and tracer variance.
        gradient_function (callable, optional): Function to compute spatial gradients.
            Defaults to compute_gradient_on_face from sva_helpers.

    Returns:
        xarray.DataArray: Divergence of advective variance flux.

    Raises:
        ValueError: If tracer, tracer variance, or horizontal velocities are not defined.
    """
    # > Get dataset
    uds = constructorSVA.ds
    # > Derive dimension names from dataset
    grid = uds.grid
    dimn_faces = grid.face_dimension
    dimn_edges = grid.edge_dimension
    dimn_layer, dimn_interfaces = get_vertical_dimensions(uds)

    # >> 0. Determine the gradient function to be used
    if gradient_function == None:
        gradient_function = compute_gradient_on_face

    # >> 1. Calculate the salinity/tracer variance 
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
            # todo calculate u1 OR completely different gradient calculation

        else:
            # > Get data in direction of the normal vector on the edges
            uh = constructorSVA.velu1
            # > Get the tracer variance
            sv2 = constructorSVA.tracer_variance
            # > If the tracer variance is not located on the edges, move from face to edge
            if dimn_edges not in sv2.dims:
                sv2 = uda_to_edges(sv2)

            else:
                pass

            # >> 3. Calculate the dot product of the velocity vector times the salinity variance (scalar)
            uhsv2 = uh.dot(sv2)

            # >> 4. Integrate the entire thing
            uhsv2_int = uhsv2.integrate(dimn_layer)
            
            # >> 5. Get the advection term after gradient calculation
            advection = gradient_function(uhsv2_int)

            return advection

def compute_straining(constructorSVA, gradient_function=None):
    """Compute the straining/mixing term (-2·w'·S'·∇S̄).

    Calculates the rate of variance production/destruction due to
    vertical mixing by internal waves and shear. Integrates over depth.

    Args:
        constructorSVA (constructorSVA): Object containing dataset, vertical velocity, and mean tracer gradient.
        gradient_function (callable, optional): Function to compute spatial gradients.
            Defaults to compute_gradient_on_face from sva_helpers.

    Returns:
        xarray.DataArray: Depth-integrated straining term.

    Raises:
        ValueError: If tracer, tracer variance, or vertical velocity is not defined.
    """
    # > Get dataset
    uds = constructorSVA.ds
    # > Derive dimension names from dataset
    grid = uds.grid
    dimn_faces = grid.face_dimension
    dimn_edges = grid.edge_dimension
    dimn_layer, dimn_interfaces = get_vertical_dimensions(uds)

    if constructorSVA.tracer_variance is None:
        if constructorSVA.tracer is None:
            raise ValueError('Please define a variable to calculate the variance of.')
        else:
            compute_tracer_variance(constructorSVA)
    if constructorSVA.velz is None:
        raise ValueError('Please define a horizontal velocity in z direction by defining constructorSVA.velz.')

    uv = constructorSVA.velz
    uv_mean = constructorSVA.velz.mean(dimn_layer)
    uv_prime = uv - uv_mean

    sv = constructorSVA._tracer_perturbation
    s_mean = constructorSVA._mean_tracer

    if dimn_edges not in s_mean:
        s_mean = uda_to_edges(s_mean)
    grad_s_mean = gradient_function(s_mean)

    straining_pt1 = -2 * uv_prime * sv
    straining = straining_pt1.dot(grad_s_mean)
    straining_int = straining.integrate(dimn_layer)

    return straining_int

def compute_dissipation(constructorSVA):
    """Compute the diffusive dissipation term (2·κ_zz·(∂S/∂z)²).

    Calculates the rate of variance dissipation by vertical diffusion
    (turbulent mixing). Integrates over depth.

    Args:
        constructorSVA (constructorSVA): Object containing dataset, tracer, and vertical diffusivity (kzz).

    Returns:
        xarray.DataArray: Depth-integrated dissipation term.

    Raises:
        ValueError: If tracer, tracer variance, or vertical diffusivity (kzz) is not defined.
    """
    # > Get dataset
    uds = constructorSVA.ds
    # > Derive dimension names from dataset
    grid = uds.grid
    dimn_faces = grid.face_dimension
    dimn_edges = grid.edge_dimension
    dimn_layer, dimn_interfaces = get_vertical_dimensions(uds)

    # > See if all required variables are given/available
    if constructorSVA.ds.tracer_variance is None:
        if constructorSVA.tracer is None:
            raise ValueError('Please define a variable to calculate the variance of.')
        else:
            compute_tracer_variance(constructorSVA)
        if constructorSVA.kzz is None:
            raise ValueError('Please define the vertical diffusion.')

    # > In D-FLOW: kzz = uda.mesh2d_vicwwu/0.7 + 5e-5 + (1/700)*1e-6
    # > Get the vertical diffusion
    kzz = constructorSVA.kzz
    # > Differentiate the tracer over the depth (dimn_interface --> dimn_layer)
    ds_dz = constructorSVA.tracer.differentiate(coord=dimn_layer)
    # > The tracer will have a different amount of depth-layers, so to match
    # > We'll have
    dissipation = 2 * kzz * ds_dz**2

    dissipation_int = dissipation.integrate(coord=dimn_layer)

    return dissipation_int
