import xugrid as xu

def reconstruct_vector_form(uds, vectors_list):
    
    # vectors_list = [uds[f'{gridname}_ucx'], uds[f'{gridname}_ucy']] # example with velocities

    gridname = uds.grid.name
    dimn_cartesian = f'{gridname}_nCartesian_coords'
    vector_name = f'{gridname}_uc'
    
    # > Define the to-be-changed attributes
    attr_list = {'standard_name' : 'sea_water_velocity', 'long_name':'Flow element center velocity vector'}
    uds[f'{vector_name}'] = xu.concat(vectors_list, dim=dimn_cartesian).assign_attrs(attr_list)

    uds = uds.drop_vars(vectors_list)

    return