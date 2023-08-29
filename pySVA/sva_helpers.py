import xugrid as xu
import xarray as xr
import warnings

def reconstruct_vector_form(constructorSVA, vectors_list, **kwargs):
    
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
        attr_list = {'standard_name' : 'sea_water_velocity', 'long_name':'Flow element center velocity vector'}
        uds[f'{vector_name}'] = vector_data = xu.concat(vectors_list_ds, dim=dimn_cartesian).assign_attrs(attr_list).rename(vector_name)

        # > Get the names of the original vectors and drop these from the dataset
        original_vector_names = vectors_list
        uds.drop_vars(original_vector_names)
    
    elif all(isinstance(i, xu.core.wrap.UgridDataArray) for i in vectors_list):

        # > Define the to-be-changed attributes
        attr_list = {'standard_name' : 'sea_water_velocity', 'long_name':'Flow element center velocity vector'}
        uds[f'{vector_name}'] = vector_data = xu.concat(vectors_list, dim=dimn_cartesian).assign_attrs(attr_list).rename(vector_name)

        # > Get the names of the original vectors and drop these from the dataset
        original_vector_names = [v.name for v in vectors_list]
        
        uds.drop_vars(original_vector_names)
    
    elif all(isinstance(i, xr.DataArray) for i in vectors_list):
        
        # > Define the to-be-changed attributes
        attr_list = {'standard_name' : 'sea_water_velocity', 'long_name':'Flow element center velocity vector'}
        uds[f'{vector_name}'] = vector_data = xr.concat(vectors_list, dim=dimn_cartesian).assign_attrs(attr_list).rename(vector_name)

        # > Get the names of the original vectors and drop these from the dataset
        original_vector_names = [v.name for v in vectors_list]
        
        uds.drop_vars(original_vector_names)
    
    else:
        warnings.warn('Type of the provided vectors_list is not recognized. Check your inputs.', UserWarning)
        return

    return vector_data