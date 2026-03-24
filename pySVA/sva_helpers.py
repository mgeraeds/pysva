"""Helper functions for pySVA.

This module contains utility functions for grid operations, coordinate calculations,
and data transformations used in salinity variance analysis.
"""

import xugrid as xu
import xarray as xr
import warnings
import numpy as np
import xarray_einstats
from functools import wraps

def deprecated(reason, version, removal):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated since {version} "
                f"and will be removed in {removal}. {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def build_inverse_distance_weights(a, b):
    """Calculate inverse distance weights using xarray.

    Computes weights for interpolation based on inverse distances
    between coordinate sets a and b. Used for mapping data between
    different grid locations (e.g., faces to edges).

    Args:
        a (xarray.DataArray): Reference coordinates (N, 2) with x and y components.
        b (xarray.DataArray): Target coordinates per reference point (N, M, 2).

    Returns:
        xarray.DataArray: Inverse distance weights (N, M) normalized to sum to 1.

    Notes:
        Weights sum to 1.0 along the second dimension for each reference point.
    """
    def weight_func(a, b):
        distance = np.linalg.norm(a[:, np.newaxis, :] - b, axis=-1)
        weights = distance / np.nansum(distance, axis=1)[:, np.newaxis] # remove this if you only want the distance 
        return weights
    
    weights = xr.apply_ufunc(weight_func, a, b,
                            input_core_dims=[list(a.dims), list(b.dims)],
                            output_core_dims=[[a.dims[0], next(iter(set(b.dims) - set(a.dims)))]])
    return weights 


def build_edge_node_connectivity(constructorSVA):
    """Build edge-node connectivity array from unstructured grid.

    Extracts and formats the edge-node connectivity information from
    an xugrid dataset, creating a properly dimensioned DataArray with
    CF-compliant attributes.

    Args:
        constructorSVA (constructorSVA): Object containing UgridDataset.

    Returns:
        xarray.DataArray: Edge-node connectivity with dimensions (n_edges, max_edge_nodes).

    Raises:
        IOError: If dataset is not a UgridDataset.
    """

    # First check if the provided dataset is a xu.core.wrap.UgridDataset
    uds = constructorSVA.ds

    if isinstance(uds, xu.core.wrap.UgridDataset):
        # > Get fill value, grid name and dimensions
        fill_value = uds.grid.fill_value
        gridname = uds.grid.name
        dimn_edges = uds.grid.edge_dimension

        # > Get coordinate names
        coord_edge_x, coord_edge_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()

        # > Determine dimension name
        dimn_maxen = f'{gridname}_nMax_edge_nodes'

        # > Get connectivity
        edge_nodes = uds.grid.edge_node_connectivity

        # > Make into xr.DataArray with correct sizes, dimensions, and coordinates
        edge_node_connectivity = xr.DataArray(data=edge_nodes, dims=[dimn_edges, dimn_maxen], coords={f'{coord_edge_x}':([dimn_edges], uds[f'{coord_edge_x}']), f'{coord_edge_y}':([dimn_edges], uds[f'{coord_edge_y}'])}, attrs={'cf_role': 'edge_node_connectivity', 'start_index':0, '_FillValue':fill_value}, name=uds.grid.to_dataset().mesh2d.attrs['edge_node_connectivity'])

    else:
        raise IOError("Please provide xu.core.wrap.UgridDataset to be able to automatically derive connectivities of the unstructured grid.")

    return edge_node_connectivity

def build_face_edge_connectivity(constructorSVA):
    """Build face-edge connectivity array from unstructured grid.

    Extracts and formats the face-edge connectivity information from
    an xugrid dataset, creating a properly dimensioned DataArray with
    CF-compliant attributes.

    Args:
        constructorSVA (constructorSVA): Object containing UgridDataset.

    Returns:
        xarray.DataArray: face_edge connectivity with dimensions (n_faces, 2).

    Raises:
        IOError: If dataset is not a UgridDataset.
    """
    # First check if the provided dataset is a xu.core.wrap.UgridDataset
    uds = constructorSVA.ds

    if isinstance(uds, xu.core.wrap.UgridDataset):
        # > Get fill value, grid name, and dimensions
        fill_value = uds.grid.fill_value
        gridname = uds.grid.name
        dimn_faces = uds.grid.face_dimension
        dimn_maxfn = uds.grid.to_dataset().mesh2d.attrs['max_face_nodes_dimension']

        # > Get voordinate names
        coord_face_x, coord_face_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()

        # > Get connectivity
        face_edges = uds.grid.face_edge_connectivity

        # > Make into xr.DataArray with correct sizes, dimensions, and coordinates
        face_edge_connectivity = xr.DataArray(face_edges, dims=[dimn_faces, dimn_maxfn],
                                              coords={f'{coord_face_x}': ([dimn_faces], uds[f'{coord_face_x}']),
                                                      f'{coord_face_y}': ([dimn_faces], uds[f'{coord_face_y}'])},
                                              attrs={'cf_role': 'face_edge_connectivity', 'start_index': 0,
                                                     '_FillValue': fill_value}, name=f'{gridname}_face_edges')
    else:
        raise IOError("Please provide xu.core.wrap.UgridDataset to be able to automatically derive connectivities of the unstructured grid.")

    return face_edge_connectivity


def get_all_coordinates(uds):

    if isinstance(uds, xu.core.wrap.UgridDataset):
        # > Get coordinate names
        coord_face_x, coord_face_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()
        coord_edge_x, coord_edge_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()
        coord_node_x, coord_node_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].split()

        # > Get dimension names
        dimn_faces = uds.grid.face_dimension
        dimn_nodes = uds.grid.node_dimension
        dimn_edges = uds.grid.edge_dimension

        # > Get grid name
        gridname = uds.grid.name

        # > Get face coordinates
        face_array = np.c_[uds.mesh2d_face_x, uds.mesh2d_face_y] # is NOT equal to uds.grid.face_coordinates
        face_coords = xr.DataArray(data=face_array, dims=[dimn_faces,f'{gridname}_nCartesian_coords'], coords={f'{coord_face_x}':([dimn_faces], uds[f'{coord_face_x}']), f'{coord_face_y}':([dimn_faces], uds[f'{coord_face_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh face', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})

        # > Get edge coordaintes
        edge_array = uds.grid.edge_coordinates # np.c_[uds.mesh2d_edge_x, uds.mesh2d_edge_y]
        edge_coords = xr.DataArray(data=edge_array, dims=[dimn_edges,f'{gridname}_nCartesian_coords'], coords={f'{coord_edge_x}':([dimn_edges], uds[f'{coord_edge_x}']), f'{coord_edge_y}':([dimn_edges], uds[f'{coord_edge_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh face', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})

        # > Get node coordinates
        node_array =  uds.grid.node_coordinates # np.c_[uds.mesh2d_node_x, uds.mesh2d_node_y]
        node_coords = xr.DataArray(data=node_array, dims=[dimn_nodes,f'{gridname}_nCartesian_coords'], coords={f'{coord_node_x}':([dimn_nodes], uds[f'{coord_node_x}']), f'{coord_node_y}':([dimn_nodes], uds[f'{coord_node_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh node', 'bounds': 'mesh2d_node_x_bnd, mesh_node_y_bnd'})
    
    elif isinstance(uds, xu.core.wrap.UgridDataArray):
        # > Get coordinate names
        coord_face_x, coord_face_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()
        coord_edge_x, coord_edge_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()
        coord_node_x, coord_node_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].split()

        # > Get dimension names
        dimn_faces = uds.grid.face_dimension
        dimn_nodes = uds.grid.node_dimension
        dimn_edges = uds.grid.edge_dimension

        # > Get grid name
        gridname = uds.grid.name
        
        # > Get face coordinates
        face_array = uds.grid.face_coordinates
        face_coords = xr.DataArray(data=face_array, dims=[dimn_faces,f'{gridname}_nCartesian_coords'], coords={f'{coord_face_x}':([dimn_faces], uds[f'{coord_face_x}']), f'{coord_face_y}':([dimn_faces], uds[f'{coord_face_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh face', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})

        # > Get edge coordaintes
        edge_array = uds.grid.edge_coordinates # np.c_[uds.mesh2d_edge_x, uds.mesh2d_edge_y]
        edge_coords = xr.DataArray(data=edge_array, dims=[dimn_edges,f'{gridname}_nCartesian_coords'], coords={f'{coord_edge_x}':([dimn_edges], uds[f'{coord_edge_x}']), f'{coord_edge_y}':([dimn_edges], uds[f'{coord_edge_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh face', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})

        # > Get node coordinates
        node_array =  uds.grid.node_coordinates # np.c_[uds.mesh2d_node_x, uds.mesh2d_node_y]
        node_coords = xr.DataArray(data=node_array, dims=[dimn_nodes,f'{gridname}_nCartesian_coords'], coords={f'{coord_node_x}':([dimn_nodes], uds[f'{coord_node_x}']), f'{coord_node_y}':([dimn_nodes], uds[f'{coord_node_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh node', 'bounds': 'mesh2d_node_x_bnd, mesh_node_y_bnd'})
    else:
        raise IOError("Please provide xu.core.wrap.UgridDataset to be able to automatically derive connectivities of the unstructured grid.")

    return face_coords, edge_coords, node_coords


def calculate_unit_normal_vectors(constructorSVA, **kwargs):
    
    # First check if the provided dataset is a xu.core.wrap.UgridDataset
    uds = constructorSVA.ds

    if isinstance(uds, xu.core.wrap.UgridDataset):
        # > Get dimensions, gridname, and coordinates
        dimn_edges = uds.grid.edge_dimension
        dimn_nodes = uds.grid.node_dimension
        fill_value = uds.grid.fill_value
        gridname = uds.grid.name
        dimn_maxen = f'{gridname}_nMax_edge_nodes'
        dimn_cart = f'{gridname}_nCartesian_coords'

        varname_unvs = f'{gridname}_unvs'

        # > See if flow area is already in the variables
        try:
            unvs = constructorSVA._unvs
            print(f'Unit normal vectors on edges already present Dataset in variable {unvs.name}.')

        except:
            if varname_unvs in uds.variables:
                print(f'Unit normal vectors on edges already present Dataset in variable {varname_unvs}.')
                return uds[varname_unvs]
            else:
                # > Get edge coordinate names
                coord_edge_x, coord_edge_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()

                # > Get kwargs
                edge_node_coords = kwargs.get('edge_node_coords')
                # face_edges = kwargs.get('face_edges')

                # > See if node-edge connectivity is given as a kwarg. If not, reconstruct it
                if 'edge_node_coords' in kwargs:
                    pass
                else:
                    # > Get the node-edge connectivity
                    edge_nodes = build_edge_node_connectivity(constructorSVA)

                    # > Build the node_coords
                    _, _, node_coords = get_all_coordinates(uds)  # just get the node_coords

                    # > Get the coordinates of all nodes belonging to an edge
                    edge_node_coords = xr.where(edge_nodes != fill_value, node_coords.isel({dimn_nodes: edge_nodes}), np.nan)

                x1 = edge_node_coords[:, 0, 0]
                x2 = edge_node_coords[:, 1, 0]
                y1 = edge_node_coords[:, 0, 1]
                y2 = edge_node_coords[:, 1, 1]
                x = x2 - x1
                y = y2 - y1

                nf = nf = xr.concat([-y, x], dimn_cart).T #np.dstack([-y, x])
                vm_nf = nf.linalg.norm(dims=dimn_cart)

                # > Calculate the norm and divide by the norm
                unvs = nf / vm_nf 
                uds[f'{varname_unvs}'] = constructorSVA._unvs = unvs.rename(varname_unvs)
    else:
        unvs = None
        
    return unvs

def reconstruct_vector_form_magnitude(constructorSVA, varname, **kwargs):
    '''Function to reconstruct the vector form of a magnitude variable on an unstructured grid that is defined on the edges
    in the direction of the normal vector (like velocity magnitude)

    :param uds:
    :param varname:
    :param kwargs:
    :return:
    '''

    # First check if the provided dataset is a xu.core.wrap.UgridDataset
    uds = constructorSVA.ds

    if isinstance(uds, xu.core.wrap.UgridDataset):
        # 1. >> Get the basics
        gridname = uds.grid.name
        dimn_faces = uds.grid.face_dimension
        dimn_edges = uds.grid.edge_dimension
        dimn_maxef = f'{gridname}_nMax_edge_faces'

        # 1. >> Get the fill value
        fill_value = uds.grid.fill_value

        # 2. >> Get basic connectivities first, if not provided in the kwargs
        # 2.1 > Get the face-edge connectivity
        if 'face_edges' in kwargs:
            face_edges = kwargs['face_edges']
        else:
            face_edges = build_face_edge_connectivity(constructorSVA)

        # > 2.2 Get the edge-face connectivity
        if 'edge_faces' in kwargs:
            edge_faces = kwargs['edge_faces']
        else:
            edge_faces = xr.DataArray(uds.grid.edge_face_connectivity, dims=(dimn_edges, dimn_maxef))

        # 3. >> Get the unit normal vectors, if not provided in the kwargs
        # 3.1. > Check if the kwargs indicate a varname for the unit normal vectors
        if 'varname_unvs' in kwargs:
            varname_unvs = kwargs['varname_unvs']
        else:
            try:
                varname_unvs = kwargs['unvs'].name
            except:
                varname_unvs = f'{gridname}_unvs'

        # 3.2. > Calculate the unit normal vectors if not in the dataset already
        try:
            unvs = uds[varname_unvs]
        except:
            # > If not in the dataset, check if it's in the kwargs
            try:
                unvs = kwargs['unvs']
            # > If not in the kwargs, then calculate it anew
            except:
                unvs = calculate_unit_normal_vectors(constructorSVA)

        # > Fill the face-edges matrix with the varname
        magn_var = uds[f'{varname}'].isel({dimn_edges: face_edges})

        # >> Determine if we're looking at a velocity value u1 or u0
        # > Because these are vector quantities in the direction of the normal vector,
        # > to get to the final vector, we have to multiply u1/u0 by the normal vector
        # > first.
        if varname == f'{gridname}_u1' or f'{gridname}_u0':
            # > These two variables are the velocity on edges in the direction of the
            # > normal vector on the current timestep (u1) and the previous timestep
            # > (u0). These are magnitudes in the directions of the normal vector, but
            # > need to be multiplied with the unit normal vector to get to a vector
            # > quantity again.

            # >> So, first we determine the sign of the velocities.
            # > For this we have to get the mesh2d_nFaces numbering of the 0th column
            # > in edge_faces (from column 0 -> 1 is positive)
            pos_fe = xr.where(face_edges != fill_value, edge_faces.isel({dimn_maxef: 0}).isel({dimn_edges: face_edges}),
                              fill_value)

            # > If the number of the 0th column in edge_faces == mesh2d_nFaces, then the
            # > direction is already positive in the right direction.
            # > Otherwise, the direction needs to be flipped
            fe_multiplier = xr.where(pos_fe == uds[dimn_faces], 1, -1)

            magn_var = magn_var * fe_multiplier

            # > Get the unit normal vectors (nf) also in the face-edges matrix
            fe_nfs = xr.where(face_edges != fill_value, unvs.isel({dimn_edges: face_edges}), np.nan)

            # > Multiply by the unit normal vector to get to a vector quantity
            # > With the multiplication we intend to calculate the dot product
            vector_var = magn_var * fe_nfs

            # > Make a new variable in the dataset
            vector_name = varname + 'c'
            constructorSVA.ds[vector_name] = constructorSVA.veluc = vector_var

            return vector_var

    else:
        raise IOError("Please provide xu.core.wrap.UgridDataset to be able to automatically derive connectivities of the unstructured grid.")


def reconstruct_vector_form(constructorSVA, vectors_list, **kwargs):
    """
    Reconstruct a vector field from its Cartesian components.

    This function concatenates scalar component fields (e.g., ``u`` and ``v``)
    into a single vector-valued DataArray along the Cartesian dimension.
    The resulting vector is stored in the underlying dataset and the original
    component variables are removed.

    Parameters
    ----------
    constructorSVA : object
        Constructor object containing the underlying UGRID dataset (``.ds``).
    vectors_list : list of str or xr.DataArray or xu.UgridDataArray
        List of vector components. Can be:
        
        - List of variable names (str) in the dataset
        - List of ``xarray.DataArray`` objects
        - List of ``xu.UgridDataArray`` objects

        All elements must be of the same type.
    **kwargs : dict, optional
        Additional keyword arguments.

        vector_name : str, optional
            Name of the reconstructed vector variable. Defaults to
            ``{gridname}_uc``.

    Returns
    -------
    xr.DataArray or xu.UgridDataArray
        Reconstructed vector field with an added Cartesian dimension.

    Raises
    ------
    UserWarning
        If the type of ``vectors_list`` is not recognized.

    Notes
    -----
    - The resulting vector is stored in ``constructorSVA.ds``.
    - Original component variables are removed from the dataset.
    - The Cartesian dimension is defined as ``{gridname}_nCartesian_coords``.
    """

    # 1. >> Get the basics
    uds = constructorSVA.ds
    gridname = uds.grid.name
    dimn_cartesian = f'{gridname}_nCartesian_coords'

    if 'vector_name' in kwargs:
        vector_name = kwargs['vector_name']
    else:
        vector_name = f'{gridname}_uc'

    # vectors_list = [uds[f'{gridname}_ucx'], uds[f'{gridname}_ucy']] # example with velocities

    # >> Case 1: if given strings in vector_list, then calculate vector notation based on the constructorSVA.ds dataset
    if all(isinstance(i, str) for i in vectors_list):

        # > Change the list to list of xu.core.wrap.UgridDataArray
        vectors_list_ds = [uds[f'{varname}'] for varname in vectors_list]

        # > Define the to-be-changed attributes
        attr_list = {'standard_name': 'sea_water_velocity', 'long_name': 'Flow element center velocity vector'}
        uds[f'{vector_name}'] = vector_data = xu.concat(vectors_list_ds, dim=dimn_cartesian).assign_attrs(
            attr_list).rename(vector_name)

        # > Get the names of the original vectors and drop these from the dataset
        original_vector_names = vectors_list
        uds.drop_vars(original_vector_names)

    elif all(isinstance(i, xu.core.wrap.UgridDataArray) for i in vectors_list):

        # > Define the to-be-changed attributes
        attr_list = {'standard_name': 'sea_water_velocity', 'long_name': 'Flow element center velocity vector'}
        uds[f'{vector_name}'] = vector_data = xu.concat(vectors_list, dim=dimn_cartesian).assign_attrs(
            attr_list).rename(vector_name)

        # > Get the names of the original vectors and drop these from the dataset
        original_vector_names = [v.name for v in vectors_list]

        uds.drop_vars(original_vector_names)

    elif all(isinstance(i, xr.DataArray) for i in vectors_list):

        # > Define the to-be-changed attributes
        attr_list = {'standard_name': 'sea_water_velocity', 'long_name': 'Flow element center velocity vector'}
        uds[f'{vector_name}'] = vector_data = xr.concat(vectors_list, dim=dimn_cartesian).assign_attrs(
            attr_list).rename(vector_name)

        # > Get the names of the original vectors and drop these from the dataset
        original_vector_names = [v.name for v in vectors_list]

        uds.drop_vars(original_vector_names)

    else:
        warnings.warn('Type of the provided vectors_list is not recognized. Check your inputs.', UserWarning)
        return

    return vector_data

def build_edge_face_weights(constructorSVA, **kwargs):
    """
    Compute inverse-distance interpolation weights from faces to edges.

    This function constructs weights for interpolating face-centered variables
    onto edges using inverse-distance weighting based on geometric distances
    between edge centers and neighboring face centroids.

    Parameters
    ----------
    constructorSVA : object
        Constructor object containing the UGRID dataset.
    **kwargs : dict, optional
        Optional keyword arguments.

        coords : tuple of xr.DataArray, optional
            Tuple ``(face_coords, edge_coords, node_coords)`` containing
            precomputed coordinate arrays. If not provided, coordinates are
            derived from the dataset.

    Returns
    -------
    xr.DataArray
        Weights for edge-face interpolation with dimensions
        ``(n_edges, nMax_edge_faces)``.

    Notes
    -----
    - Uses edge-face connectivity to associate edges with neighboring faces.
    - Weights are computed using inverse-distance weighting.
    - Fill values in connectivity arrays are masked before computation.
    """

    uds = constructorSVA.ds

	# Get dimension names
    dimn_maxfn = uds.ugrid.grid.to_dataset().mesh2d.attrs['max_face_nodes_dimension']
    dimn_faces = uds.ugrid.grid.face_dimension
    dimn_edges = uds.ugrid.grid.edge_dimension
    fill_value = uds.ugrid.grid.fill_value
    gridname = uds.ugrid.grid.name
    dimn_maxef = f'{gridname}_nMax_edge_faces'
	
    # > Get the edge-face connectivity
    edge_faces = xr.DataArray(uds.ugrid.grid.edge_face_connectivity, dims=(dimn_edges, dimn_maxef))
    
    if 'coords' in kwargs:
        face_coords, edge_coords, _ = kwargs['coords']
    else:
        # > Get all relevant coordinates
        face_coords, edge_coords, _ = get_all_coordinates(uds)

	# > Fill edge-face-connectivity matrix with face coordinates
    edge_face_coords = xr.where(edge_faces!=fill_value, face_coords.isel({dimn_faces:edge_faces}), np.nan)
    
    # > Build the weights
    weights = build_inverse_distance_weights(edge_coords, edge_face_coords)
    
    return weights

def calculate_distance_haversine(lat, lon, lat_or, lon_or):
    """
    Compute geodesic distance between coordinates using the WGS84 ellipsoid.

    Parameters
    ----------
    lat : float or pandas.Series
        Latitude(s) of target location(s) in degrees.
    lon : float or pandas.Series
        Longitude(s) of target location(s) in degrees.
    lat_or : float
        Reference latitude in degrees.
    lon_or : float
        Reference longitude in degrees.

    Returns
    -------
    float or ndarray
        Geodesic distance(s) in meters.

    Notes
    -----
    - Uses ``pyproj.Geod.inv`` for accurate ellipsoidal distance computation.
    - Supports both scalar inputs and pandas Series.
    """

    import pandas as pd
    import pyproj

    geodesic = pyproj.Geod(ellps='WGS84')

    if type(lat) and type(lon) == pd.core.series.Series:
        lat_or_list = [lat_or for n in range(len(lat))]
        lon_or_list = [lon_or for n in range(len(lon))]

        fwd_azimuth, back_azimuth, distance = geodesic.inv(lat.tolist(), lon.tolist(), lat_or_list, lon_or_list)

    else:
        fwd_azimuth, back_azimuth, distance = geodesic.inv(lat, lon, lat_or, lon_or)

    #     x = haversine(row.ix[0],row.ix[1],lat_or,lon_or) #starting point; approx. Botlek harbour
    return distance  # pd.Series(distance)#(x)

def calculate_distance_pythagoras(x1, y1, x2, y2):
    """
    Compute Euclidean distance in a Cartesian coordinate system.

    Parameters
    ----------
    x1, y1 : float or array-like
        Coordinates of the starting point.
    x2, y2 : float or array-like
        Coordinates of the ending point.

    Returns
    -------
    float or ndarray
        Euclidean distance(s) in meters.

    Notes
    -----
    - Assumes planar (projected) coordinates.
    - Equivalent to the L2 norm of the coordinate difference.
    """

    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance

@deprecated(
    reason="Use constructorSVA.compute_gradient_on_face instead.",
    version="1.0.0",
    removal="1.2.0",
)
def compute_gradient_on_face(constructorSVA, uda, **kwargs):
    """
    Compute the gradient of a scalar field on mesh faces.

    The gradient is computed using a finite-volume formulation by integrating
    fluxes across edges and normalizing by cell volume.

    Parameters
    ----------
    constructorSVA : object
        Constructor object containing the UGRID dataset.
    uda : xr.DataArray or xu.UgridDataArray
        Input scalar field defined on edges.
    **kwargs : dict, optional
        Optional keyword arguments.

        unvs : xr.DataArray, optional
            Precomputed unit normal vectors on edges. If not provided,
            they are calculated internally.

    Returns
    -------
    xr.DataArray
        Gradient field on faces with an additional Cartesian dimension.

    Raises
    ------
    ValueError
        If the input variable is not defined on edges.
    NameError
        If required variables (flow area or volume) are missing.

    Notes
    -----
    - Uses face-edge connectivity and unit normal vectors.
    - Ensures outward-pointing normals via dot-product sign correction.
    - Result is stored in the dataset as ``{varname}_gradient``.

    .. deprecated:: 1.0.0
       This function will be removed in version 1.2.0.
       Use :meth: constructorSVA.compute_gradient_on_face` instead.
    
    """

    from dfm_tools.xugrid_helpers import get_vertical_dimensions
    
    # > Obtain uds from constructorSVA object
    uds = constructorSVA.ds

    # > Get grid
    grid = uds.grid

    # > Get dimension and grid names
    dimn_maxfn = grid.to_dataset().mesh2d.attrs['max_face_nodes_dimension']
    dimn_faces = grid.face_dimension
    dimn_edges = grid.edge_dimension
    fill_value = grid.fill_value
    gridname = grid.name
    dimn_maxef = f'{gridname}_nMax_edge_faces'
    dimn_layer, dimn_interfaces = get_vertical_dimensions(uds)
    dimn_cart = f'{gridname}_nCartesian_coords'

    # > Determine varname from the provided array\
    varname = uda.name

    # > Calculate the unit normal vectors if not in the dataset already
    try:
        unvs = kwargs['unvs']
    except:
        unvs = calculate_unit_normal_vectors(constructorSVA)
    
    # > Get the volume and flow area variables: check if they're in the 
    # > constructor first, then check the dataset, else throw error
    try:
        flow_area = constructorSVA.flow_area
    except:
        try:
            flow_area = uds[f'{gridname}_au'] # todo: extend to more arbitrary names in later stage
        except: 
            raise NameError('Could not retrieve the flow area, which is necessary for the gradient \
                             calculation. Please provide volume variable in the constructor or in the underlying dataset.')
    
    try:
        volume = constructorSVA.volume
    except:
        try:
            volume = uds[f'{gridname}_vol1']# todo: extend to more arbitrary names in later stage
        except:
            raise NameError('Could not retrieve the volume of the cells, which is necessary for the gradient \
                             calculation. Please provide volume variable in the constructor or in the underlying dataset.')

    # > Get the edge-face connectivity and replace fill values with -1
    edge_faces = xr.DataArray(uds.ugrid.grid.edge_face_connectivity, dims=(dimn_edges, dimn_maxef))
    edge_faces_validbool = edge_faces!=fill_value
    edge_faces = edge_faces.where(edge_faces_validbool, -1)
        
    # > Get the face-edge connectivity and replace fill values with -1
    face_edges = build_face_edge_connectivity(constructorSVA)
    face_edges_validbool = face_edges!=fill_value
    face_edges = face_edges.where(face_edges_validbool, -1)
    face_edges_stacked = face_edges.stack(__tmp_dim__=(dimn_faces, dimn_maxfn))

    # > Get the unit normal vectors (nf) also in the face-edges matrix
    fe_nfs = xr.where(face_edges_validbool, unvs.isel({dimn_edges:face_edges}), np.nan)

    # // 2. See if the normal vectors are pointing out of the cell. If not, flip them.
    # > Calculate distance vectors
    dv = calculate_distance_vectors(constructorSVA)
    # > Calculate the dot product between the calculated normal vectors and the distance vector for each face
    i_nfs = xr.dot(fe_nfs, dv, dims=[dimn_cart])
    # > if the product < 1, multiply by -1 to get an outwards facing normal vector, and update the variable
    fe_nfs = xr.where(i_nfs > 0, fe_nfs, fe_nfs * -1)

	# > Determine if we're looking at a velocity value u1 or u0
    # > Because these are vector quantities in the direction of the normal vector,
    # > to get to the final vector, we have to multiply u1/u0 by the normal vector
    # > first. Also, we need to check their sign for every edge.
    if dimn_edges in uda.dims:
        # > Make sure the edge dimension is not chunked, otherwise we will 
        # > get "PerformanceWarning: Slicing with an out-of-order index is generating x times more chunks."
        chunks = {dimn_edges:-1}
        uda = uda.chunk(chunks)

        # > Fill the face-edges matrix with the varname
        # > Do this via stack and unstack since 2D indexing does not
        # > properly work in dask yet: https://github.com/dask/dask/pull/10237
        edge_var_stacked = uda.isel({dimn_edges: face_edges_stacked})
        edge_var = edge_var_stacked.unstack("__tmp_dim__")
        # > Convert data-array back to an xu.UgridDataArray
        edge_var = xu.UgridDataArray(edge_var, grid=grid)
        # > Replace locations of the validbools with NaN's
        edge_var = xr.where(face_edges_validbool, edge_var, np.nan)

        # > Fill face_edge matrix with flow area data
        flow_area = flow_area.chunk(chunks)
        edge_au_stacked = flow_area.isel({dimn_edges:face_edges_stacked})
        edge_au = edge_au_stacked.unstack("__tmp_dim__")

        # > Multiply the variable with the edge area (flow area), multiply by the 
        # > "flipped boolean" and the unit normal vector, and sum (dimension: faces)
        face_vars = (edge_var * edge_au * fe_nfs).sum(dim=dimn_maxfn, keep_attrs=True)
        
        # > Multiply the total result with (1/cell volume) (dimension: faces)
        gradient = (1/volume) * face_vars
        uds[f'{varname}_gradient'] = gradient
        
        return gradient

    else: 
        raise ValueError('This function only supports the calculation of gradients at face location, \
                         based on data on edges. Please supply a variable that is located on the edges.')

@deprecated(
    reason="Compute from constructorSVA.compute_gradient_on_face() instead.",
    version="1.0.0",
    removal="1.2.0",
)
def compute_divergence_on_face(constructorSVA, uda, **kwargs):
    """
    Compute the divergence of a vector field on mesh faces.

    The divergence is calculated using a finite-volume formulation by summing
    fluxes across edges and normalizing by cell volume.

    Parameters
    ----------
    constructorSVA : object
        Constructor object containing the UGRID dataset.
    uda : xr.DataArray or xu.UgridDataArray
        Vector field defined on edges. Must be aligned with edge-normal directions.
    **kwargs : dict, optional
        Optional keyword arguments.

        unvs : xr.DataArray, optional
            Precomputed unit normal vectors.

    Returns
    -------
    xr.DataArray
        Divergence field on faces.

    Raises
    ------
    ValueError
        If the input variable is not defined on edges.

    Notes
    -----
    - Accounts for edge orientation using connectivity-based sign correction.
    - Result is stored in the dataset as ``{varname}_divergence``.

    .. deprecated:: 1.0.0
       This function will be removed in version 1.2.0.
       Use :meth: constructorSVA.compute_gradient_on_face` instead.
    """
    # For this function, the vector on the edge needs to be in the direction of the normal vector!!
    from dfm_tools.xugrid_helpers import get_vertical_dimensions
    
    # > Obtain uds from constructorSVA object
    uds = constructorSVA.ds

    # > Get grid
    grid = uds.grid

    # > Get dimension and grid names
    dimn_maxfn = grid.to_dataset().mesh2d.attrs['max_face_nodes_dimension']
    dimn_faces = grid.face_dimension
    dimn_edges = grid.edge_dimension
    fill_value = grid.fill_value
    gridname = grid.name
    dimn_maxef = f'{gridname}_nMax_edge_faces'
    dimn_layer, dimn_interfaces = get_vertical_dimensions(uds)
    dimn_cart = f'{gridname}_nCartesian_coords'

    # > Determine varname from the provided array
    varname = uda.name

    # > Calculate the unit normal vectors if not in the dataset already
    try:
        unvs = kwargs['unvs']
    except:
        unvs = calculate_unit_normal_vectors(constructorSVA)
    
    # > Get the volume and flow area variables: check if they're in the 
    # > constructor first, then check the dataset, else throw error
    try:
        flow_area = constructorSVA.flow_area
    except:
        try:
            flow_area = uds[f'{gridname}_au'] # todo: extend to more arbitrary names in later stage
        except: 
            raise NameError('Could not retrieve the flow area, which is necessary for the gradient \
                             calculation. Please provide volume variable in the constructor or in the underlying dataset.')
    
    try:
        volume = constructorSVA.volume
    except:
        try:
            volume = uds[f'{gridname}_vol1']# todo: extend to more arbitrary names in later stage
        except:
            raise NameError('Could not retrieve the volume of the cells, which is necessary for the gradient \
                             calculation. Please provide volume variable in the constructor or in the underlying dataset.')

    # > Get the edge-face connectivity and replace fill values with -1
    edge_faces = xr.DataArray(uds.ugrid.grid.edge_face_connectivity, dims=(dimn_edges, dimn_maxef))
    edge_faces_validbool = edge_faces!=fill_value
    edge_faces = edge_faces.where(edge_faces_validbool, -1)
        
    # > Get the face-edge connectivity and replace fill values with -1
    face_edges = build_face_edge_connectivity(constructorSVA)
    face_edges_validbool = face_edges!=fill_value
    face_edges = face_edges.where(face_edges_validbool, -1)
    face_edges_stacked = face_edges.stack(__tmp_dim__=(dimn_faces, dimn_maxfn))

    # > Get the unit normal vectors (nf) also in the face-edges matrix
    fe_nfs = xr.where(face_edges!=fill_value, unvs.isel({dimn_edges:face_edges}), np.nan)

    # // 2. See if the normal vectors are pointing out of the cell. If not, flip them.
    # > Calculate distance vectors
    dv = calculate_distance_vectors(constructorSVA)
    # > Calculate the dot product between the calculated normal vectors and the distance vector for each face
    i_nfs = xr.dot(fe_nfs, dv, dims=[dimn_cart])
    # > if the product < 1, multiply by -1 to get an outwards facing normal vector, and update the variable
    fe_nfs = xr.where(i_nfs > 0, fe_nfs, fe_nfs * -1)

	# > Determine if we're looking at a velocity value u1 or u0
    # > Because these are vector quantities in the direction of the normal vector,
    # > to get to the final vector, we have to multiply u1/u0 by the normal vector
    # > first. Also, we need to check their sign for every edge.
    if dimn_edges in uda.dims:
        # > Make sure the edge dimension is not chunked, otherwise we will 
        # > get "PerformanceWarning: Slicing with an out-of-order index is generating x times more chunks."
        chunks = {dimn_edges:-1}
        uda = uda.chunk(chunks)

        # > Fill the face-edges matrix with the varname
        # > Do this via stack and unstack since 2D indexing does not
        # > properly work in dask yet: https://github.com/dask/dask/pull/10237
        edge_var_stacked = uda.isel({dimn_edges: face_edges_stacked})
        edge_var = edge_var_stacked.unstack("__tmp_dim__")
        # > Convert data-array back to an xu.UgridDataArray
        edge_var = xu.UgridDataArray(edge_var, grid=grid)

		# >> We have to determine the sign of the velocity 
		# >> Determine whether the u1 value is positive or negative
		# > Get the mesh2d_nFaces numbering of the 0th column in edge_faces (from
        # >  0 -> 1 is positive)
        pos_fe = xr.where(face_edges!=fill_value, edge_faces.isel({dimn_maxef:0}).isel({dimn_edges:face_edges}), fill_value)
        pos_fe = pos_fe.where(face_edges_validbool, -1)

		# > If the number of the 0th column in edge_faces == mesh2d_nFaces, then the 
        # > direction is already positive in the right direction.
		# > Otherwise, the direction needs to be flipped
        fe_multiplier = xr.where(pos_fe==uds[dimn_faces], 1, -1)
        edge_var = edge_var * fe_multiplier

        # > Calculate the dot product of the vector quantity with the unit normal vectors
        edge_var = edge_var * fe_nfs 

        # > Fill face_edge matrix with flow area data
        flow_area = flow_area.chunk(chunks)
        edge_au_stacked = flow_area.isel({dimn_edges:face_edges_stacked})
        edge_au = edge_au_stacked.unstack("__tmp_dim__")
        
        # > Multiply the variable with the edge area (flow area), multiply by the 
        # > "flipped boolean" and the unit normal vector, and sum (dimension: faces)
        face_vars = (edge_var * edge_au).sum(dim=dimn_maxfn, keep_attrs=True)
        
        # > Multiply the total result with (1/cell volume) (dimension: faces)
        uds[f'{varname}_divergence'] = divergence = ((1/volume) * face_vars).sum(dim=dimn_cart)
        
        return divergence

    else: 
        raise ValueError('This function only supports the calculation of the divergence at face location, \
                         based on data on edges. Please supply a variable that is located on the edges.')

@deprecated(
    reason="Compute from constructorSVA.uda_to_edges() instead.",
    version="1.0.0",
    removal="1.2.0",
)
def uda_to_edges(constructorSVA, uda):
    """
    Interpolate a face-centered variable to edges.

    Performs inverse-distance weighted interpolation using edge-face
    connectivity.

    Parameters
    ----------
    constructorSVA : object
        Constructor object containing the UGRID dataset.
    uda : xu.UgridDataArray
        Input variable defined on faces.

    Returns
    -------
    xu.UgridDataArray
        Interpolated variable on edges.

    Notes
    -----
    - Uses weights from :func:`build_edge_face_weights`.
    - Missing neighbors are handled via NaN masking.
    - Result is assigned location attribute ``'edge'``.
    """
    uds = constructorSVA.ds

    # > Get grid
    grid = uda.grid

    # > Get dimension and grid names
    dimn_maxfn = grid.to_dataset().mesh2d.attrs['max_face_nodes_dimension']
    dimn_faces = grid.face_dimension
    dimn_edges = grid.edge_dimension
    fill_value = grid.fill_value
    gridname = grid.name
    dimn_maxef = f'{gridname}_nMax_edge_faces'

    # > Get the edge-face connectivity and replace fill values with -1
    edge_faces = xr.DataArray(uda.ugrid.grid.edge_face_connectivity, dims=(dimn_edges, dimn_maxef))
    edge_faces_validbool = edge_faces!=fill_value
    edge_faces = edge_faces.where(edge_faces_validbool, -1)

    # > Make sure the face dimension is not chunked, otherwise we will 
    # > get "PerformanceWarning: Slicing with an out-of-order index is generating x times more chunks."
    chunks = {dimn_faces:-1}
    uda = uda.chunk(chunks)

    # > Select the varname on faces in the edge-face connetivity matrix
    edge_faces_stacked = edge_faces.stack(__tmp_dim__=(dimn_edges, dimn_maxef))
    edge_var_stacked = uda.isel({dimn_faces: edge_faces_stacked})
    edge_var = edge_var_stacked.unstack("__tmp_dim__")
    # > Convert data-array back to an xu.UgridDataArray
    edge_var = xu.UgridDataArray(edge_var, grid=grid)

    # > Set fill values to nan-values
    edge_var = edge_var.where(edge_faces_validbool, np.nan)

    # > Calculate the variable on the edges, based on the face_weights
    face_weights = build_edge_face_weights(constructorSVA)
    edge_var = (edge_var * face_weights).sum(dim=dimn_maxef)
    # edge_var = edge_var.mean(dim=dimn_maxef)
    # Give name and attributes
    attr_list = {'location': 'edge', 'cell_methods': f'{dimn_faces}: inverse distance weighted mean'}
    edge_var = edge_var.assign_attrs(
            attr_list).rename(uda.name)
    
    return edge_var

def calculate_distance_vectors(constructorSVA, **kwargs):

    uds = constructorSVA.ds

    # > Get dimensions, fill_value, and varname
    gridname = uds.grid.name
    fill_value = uds.grid.fill_value
    dimn_edges = uds.grid.edge_dimension

    # Get  coordinate names
    coord_edge_x, coord_edge_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()
    # coord_node_x, coord_node_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].split()
    coord_face_x, coord_face_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()

    # > Get kwargs
    face_edges = kwargs.get('face_edges')

    # > Check if face-edge-connectivity is given
    if 'face_edges' in kwargs:
        pass
    else:
        face_edges = build_face_edge_connectivity(constructorSVA)

    # > Get dimension names
    dimn_faces = uds.grid.face_dimension

    # > Get the cell centroid coordinates
    centroid_array = uds.grid.face_coordinates
    centroid_coords = xr.DataArray(data=centroid_array, dims=[dimn_faces, f'{gridname}_nCartesian_coords'], coords={f'{coord_face_x}':([dimn_faces], uds[f'{coord_face_x}']), f'{coord_face_y}':([dimn_faces], uds[f'{coord_face_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh centroids', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})   

    # >> 1. Get the distance vector
    # > Get the edge coordinates
    edge_array = uds.grid.edge_coordinates # np.c_[uds.mesh2d_edge_x, uds.mesh2d_edge_y]
    edge_coords = xr.DataArray(data=edge_array, dims=[dimn_edges,f'{gridname}_nCartesian_coords'], coords={f'{coord_edge_x}':([dimn_edges], uds[f'{coord_edge_x}']), f'{coord_edge_y}':([dimn_edges], uds[f'{coord_edge_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh face', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})   

    # > Put edge coordinates into face-edge connectivity matrix
    face_edge_coords = xr.where(face_edges!=fill_value, edge_coords.isel({dimn_edges:face_edges}), np.nan)

    # > Subtract the face coordinates from each of the edge coordinates in the connectivity matrix
    distance_vectors = face_edge_coords - centroid_coords

    return distance_vectors

def compute_kzz(uds, dicoww=5e-5, prandtl_schmidt=0.7,  tracer='mesh2d_sa1'):

    gridname = uds.grid.name

    vicwwu = uds[f'{gridname}_vicwwu']
    dicwwu = vicwwu / prandtl_schmidt
    
    if tracer == f'{gridname}_sa1':
        k_l = (1/700) * 10e-6
    elif tracer == f'{gridname}_tem1': 
        k_l = (1/6.7) * 10e-6

    uds['mesh2d_dicwwu'] = dicwwu + dicoww + k_l

    return uds

import xarray as xr
from typing import Optional

def integrate_trapz(y: xr.DataArray, x: xr.DataArray, dim: Optional[str] = None) -> xr.DataArray:
    """
    Perform trapezoidal integration along a specified dimension.

    Parameters:
        y (xr.DataArray): The values to integrate.
        x (xr.DataArray): The coordinate values along which to integrate.
        dim (str, optional): The dimension along which to integrate. If None, the operation
                             is applied along all dimensions.

    Returns:
        xr.DataArray: The result of the trapezoidal integration.
    """
    # First drop the x coordinate (the one to integrate over), as this will cause conflicts
    y = y.drop_vars(x.name)
    x = x.drop_vars(x.name)
    
    desired_order = list(x.dims)
    y = y.transpose(*desired_order)
    
    # Calculate dx and average y
    dx = x.isel({dim: slice(1, None)}) - x.isel({dim: slice(None, -1)})
    avg_y = (y.isel({dim: slice(1, None)}) + y.isel({dim: slice(None, -1)})) / 2

    result = (dx * avg_y).sum(dim=dim)

    return result

def depth_int2volume_int(uda: xr.DataArray, cell_area: xr.DataArray, dimn_faces: str) -> xr.DataArray:
    """
    Convert a depth-integrated DataArray to a volume-integrated DataArray.

    Parameters:
    ----------
    uda : xr.DataArray
        The input DataArray representing depth-integrated values.
    cell_area : xr.DataArray
        A 2D DataArray representing the cell area for each spatial grid point.
    dimn_faces : str
        The name of the faces dimension to integrate over.

    Returns:
    -------
    xr.DataArray
        The volume-integrated DataArray.
    """
    volume_int = (uda * cell_area).sum(dim=[dimn_faces])
    
    return volume_int

def differentiate_over_3d_coord(uda: xr.DataArray, coord_var: str, axis: int = -1) -> xr.DataArray:
    """
    Differentiate a Dask-backed xarray DataArray over a 3D coordinate variable.

    Parameters:
    ----------
    uda : xr.DataArray
        The input DataArray to differentiate.
    coord_var : str
        The name of the 3D coordinate variable.
    axis : int, optional
        The axis number of the coordinate dimension to differentiate over. Defaults to -1.
        
    Returns:
    -------
    xr.DataArray
        Differentiated DataArray over the given coordinate.
    """
    # Deduce name of depth dimension
    depth_dim = uda[coord_var].dims[axis]
    
    # Calculate central difference with .diff() to reduce shifts
    # data_diff = uda.diff(depth_dim)
    # coord_diff = uda[coord_var].diff(depth_dim)
    
    # Calculate the derivative and re-align dimensions with padding
    # differentiated = data_diff / coord_diff
    differentiated = xr.apply_ufunc(
        lambda x, y: x.diff(depth_dim) / y.diff(depth_dim),
        uda, uda[coord_var],
        dask="parallelized",
        output_dtypes=[uda.dtype],
        join="override"
    )

    differentiated = differentiated.pad({depth_dim: (1, 0)}, mode="edge")

    return differentiated
