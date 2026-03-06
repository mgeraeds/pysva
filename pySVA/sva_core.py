"""Core module for pySVA containing the main constructorSVA class.

This module defines the constructorSVA class which handles data loading,
grid operations, and provides access to various hydrodynamic properties
for salinity variance analysis.
"""

__all__ = ['constructorSVA']

import xarray as xr
import xugrid as xu
import dfm_tools as dfmt
import dask.array as da
import numpy as np
from functools import cached_property
from pySVA.sva_helpers import build_inverse_distance_weights, integrate_trapz, calculate_distance_pythagoras, differentiate_over_3d_coord

class constructorSVA:
    """Main class for Salinity Variance Analysis (SVA).

    Loads and processes hydrodynamic data from D-Flow FM simulations,
    providing access to grid properties, velocities, tracers, and
    computed quantities needed for variance budget analysis.

    The class handles multiple input formats (netCDF files, xarray Datasets,
    or xugrid UgridDatasets) and automatically extracts grid topology and
    coordinate systems.

    Attributes:
        ds (xarray.Dataset): The loaded hydrodynamic dataset.
        grid: The unstructured grid object.
        Various dimension names (dimn_edges, dimn_faces, etc.) and coordinates.
    """

    def __init__(self, input_file, data_description, **kwargs):
        """Initialize the SVA analysis object.

        Args:
            input_file (str or xr.Dataset or xu.UgridDataset or xr.DataArray):
                Path to netCDF file, pre-loaded xarray Dataset, UgridDataset,
                or DataArray containing hydrodynamic data.
            data_description (dict): Mapping of variable names to dataset keys.
                Required keys: 'velx', 'vely', 'velz', 'viscosity', 'tracer', 'depth'.
                Optional keys: 'horizontal_diffusivity', 'interfaces', 'bed_level',
                'volume', 'flow_area'.
            **kwargs: Additional arguments passed to dfmt.open_partitioned_dataset().

        Raises:
            IOError: If file cannot be read or input type is not supported.
        """

        if isinstance(input_file, str):
            try:
                self._read(file_name=input_file)
            except (OSError, IOError, RuntimeError):
                raise IOError("Unable to read file.")
        elif isinstance(input_file, xr.Dataset):
            self.ds = input_file
        elif isinstance(input_file, xu.core.wrap.UgridDataset):
            self.ds = input_file
        elif isinstance(input_file, xr.DataArray):
            self.ds = input_file
        else:
            raise IOError("Please provide xr.Dataset, xu.core.wrap.UgridDataset, xr.DataArray, or a file path to a netCDF.")

        # Dimension-related attributes
        self.grid = self.ds.grid
        self.gridname = self.ds.grid.name
        self.dimn_maxef = f'{self.gridname}_nMax_edge_faces'
        self.dimn_maxfn = self.grid.to_dataset().mesh2d.attrs['max_face_nodes_dimension']
        self.dimn_maxen = f'{self.gridname}_nMax_edge_nodes'
        self.dimn_cart = f'{self.gridname}_nCartesian_coords'

        self.dimn_nodes = self.ds.grid.node_dimension
        self.dimn_edges = self.ds.grid.edge_dimension
        self.dimn_faces = self.ds.grid.face_dimension
        self.fill_value = self.ds.grid.fill_value
        self.dimn_layer = self.ds.grid.to_dataset()[self.gridname].layer_dimension
        self.dimn_interface = self.ds.grid.to_dataset()[self.gridname].interface_dimension

        self.face_coords, self.edge_coords, self.node_coords = self.get_all_coordinates()

        # Mandatory attributes
        self.velx = self.ds[f'{data_description["velx"]}']
        self.vely = self.ds[f'{data_description["vely"]}']
        self.velz = self.ds[f'{data_description["velz"]}']
        self.viscosity = self.ds[f'{data_description["viscosity"]}']
        self.tracer = self.ds[f'{data_description["tracer"]}']
        self.depth = self.ds[f'{data_description["depth"]}']
        
        if 'horizontal_diffusivity' in data_description.keys():
            self.diu = self.ds[f'{data_description["horizontal_diffusivity"]}']
        else: # > Todo: add possible calculation of these
            self.diu = None
            
        if 'interfaces' in data_description.keys():
            self.interfaces = self.ds[f'{data_description["interfaces"]}']
        else: # > Todo: add possible calculation of these
            self.interfaces = None
            
        if 'bed_level' in data_description.keys():
            self.bed_level = self.ds[f'{data_description["bed_level"]}']
        else:
            try: 
                self.bed_level = self.bed_level
            except:
                self.bed_level = None 
                
        # Optional attributes
        if 'volume' in data_description.keys():
            self.volume = self.ds[f'{data_description["volume"]}']
        else:
            try:
                self.volume = self.volume
            except:
                self.volume = None
        
        if 'flow_area' in data_description.keys():
            self.flow_area = self.ds[f'{data_description["flow_area"]}']
        else:
            self.flow_area = None #self.flow_area
#             try:
#                 self.flow_area = self.flow_area
#             except:
#                 self.flow_area = None  
                
        # Hidden/calculated values
        self.edge_nodes = self.edge_nodes
        self.face_edges = self.face_edges
        self.kzz = self.kzz
        self.tracer_variance = self.tracer_variance

    def _read(self, file_name, **kwargs):
        """Read hydrodynamic dataset from file.

        Loads a partitioned netCDF dataset using dfm_tools.

        Args:
            file_name (str): Path to the netCDF file.
            **kwargs: Additional arguments passed to dfmt.open_partitioned_dataset().
        """
        # self.ds = xr.open_dataset(file_name, use_cftime=True, **kwargs)
        self.ds = dfmt.open_partitioned_dataset(file_name, **kwargs)
            
#     @cached_property
#     def flow_area(self):
#         # > Get dimension names
#         dimn_faces = self.dimn_faces
#         dimn_maxef = self.dimn_maxef
#         dimn_interface = self.dimn_interface
#         dimn_edges = self.dimn_edges
        
#         # > Get general strings
#         fill_value = self.fill_value
#         gridname = self.gridname
        
#         # > Get properties
#         bed_level = self.bed_level
#         interfaces = self.interfaces
#         dimn_layer = self.dimn_layer
#         edge_faces = self.edge_faces
        
#         # > Get the edge-face connectivity and replace fill values with -1
#         edge_faces = self.edge_faces
#         edge_faces = edge_faces.reset_coords(drop=True)
#         edge_faces_validbool = (edge_faces!=fill_value).data
#         edge_faces = edge_faces.where(edge_faces_validbool, -1)

#         # > Make sure the face dimension is not chunked, otherwise we will 
#         # > get "PerformanceWarning: Slicing with an out-of-order index is generating x times more chunks."
#         chunks = {dimn_faces:-1}
#         bed_level = bed_level.chunk(chunks)
#         interfaces = interfaces.chunk(chunks)

#         # > Select the varname on faces in the edge-face connectivity matrix
#         edge_faces_stacked = edge_faces.stack(__tmp_dim__=(dimn_edges, dimn_maxef))

#         # > // Get bed levels at nodes and bed levels at edges //
#         # > Get the bed_levels on the faces, and fill it in in the edge_node_connectivity matrix 
#         bl_edge_faces_stacked = bed_level.isel({dimn_faces: edge_faces_stacked})
#         bl_edge_faces = bl_edge_faces_stacked.unstack("__tmp_dim__")
#         bl_edge_faces = xr.where(edge_faces_validbool, bl_edge_faces, np.nan)

#         # // Get face-based water depth (Algorithm 4, Equation 6.14) //
#         # > Get the water levels on the faces, and fill it in in the edge-face connectivity matrix
#         wl_edge_faces_stacked = interfaces.isel({dimn_faces:edge_faces_stacked})
#         wl_edge_faces = wl_edge_faces_stacked.unstack("__tmp_dim__")
#         wl_edge_faces = xr.where(edge_faces_validbool, wl_edge_faces, np.nan)

#         # > Calculate the water depth based on Algorithm 4 of the Technical refernece manual (huj in Eq. 6.14)
# #         wd_edges = xr.where(wl_edge_faces[:,0,:,:] > wl_edge_faces[:,1,:,:], wl_edge_faces[:,0,:,:] - bl_edge_faces.min(dim=dimn_maxef), wl_edge_faces[:,1,:,:] - bl_edge_faces.min(dim=dimn_maxef)).diff(dim=dimn_interface)
#         bl_min = bl_edge_faces.min(dim=dimn_maxef)
#         wl0 = wl_edge_faces.isel({dimn_maxef: 0})
#         wl1 = wl_edge_faces.isel({dimn_maxef: 1})
#         wd_edges = xr.where(wl0 > wl1, wl0 - bl_min, wl1 - bl_min).diff(dim=dimn_interface)

#         # // Calculate the cross-sectional bed variation //
#         # > Calculate the difference between the bed levels (6.16)
#         delta_bl = bl_edge_faces.max(dim=dimn_maxef) - bl_edge_faces.min(dim=dimn_maxef)
#         # > Define delta value
#         delta = 10e-3

#         # > Get the edge_length
#         edge_length = self.edge_length
        
#         # > Get minimum of huj/delta_blj and 1 
#         min_huj_bl = xr.where(wd_edges/edge_length < 1, wd_edges/edge_length, 1)
#         # > Turn nan's back into min_huj_bl
#         min_huj_bl = xr.where((wd_edges/edge_length)!=np.nan, min_huj_bl, np.nan)
#         # Get minumum of delta_blj/huj
#         min_bl_huj = xr.where(edge_length/wd_edges < 1, edge_length/wd_edges, 1)
#         # > Turn nan's back into min_bl_huj
#         min_bl_huj = xr.where((edge_length/wd_edges)!=np.nan, min_bl_huj, np.nan) 
        
#         # Chunk back
#         bed_level = bed_level.chunk("auto")
#         interfaces = interfaces.chunk("auto")

#         # > Calculate actual flow area based on Algorithm 5 (Eqs 6.16 and 6.17)
#         flow_area = xr.where(delta_bl < delta*edge_length, edge_length*wd_edges, edge_length*wd_edges*min_huj_bl*(1-0.5*min_bl_huj)).rename({dimn_interface:dimn_layer})
#         flow_area = flow_area.rename(f'{gridname}_au')
        
#         return flow_area
   
    @cached_property
    def face_area(self):
        """Calculate face area for each grid edge.

        Computes the area of grid faces by multiplying cell thickness
        (interpolated to edges) by edge length.

        Returns:
            xarray.DataArray: Face area with dimensions matching edges.
        """
        dimn_edges = self.dimn_edges
        
        # > Get general strings
        fill_value = self.fill_value
        gridname = self.gridname
        
        # > Get edge length
        edge_length = self.edge_length
        edge_length = edge_length
        # > Get the cell thickness 
        cell_thickness = self.cell_thickness

        # > Interpolate the cell thickness on the edges 
        cell_thickness_edges = self.uda_to_edges(cell_thickness)
        
        chunks = {dimn_edges:-1}
        cell_thickness_edges = cell_thickness_edges.chunk(chunks)
        
        face_area = (cell_thickness_edges * edge_length).rename(f'{gridname}_face_area')
        face_area = face_area.chunk('auto')
        
        return face_area
    
    @cached_property
    def water_depth(self):
        """Calculate water depth at each grid face.

        Computes depth as the difference between maximum interface elevation
        (water surface) and bed level.

        Returns:
            xarray.DataArray: Water depth (positive downward) at each face.
        """
        
        # > Get general strings
        gridname = self.gridname
        
        # > Get dimension names
        dimn_interface = self.dimn_interface
        
        # > Get relevant properties
        interfaces = self.interfaces
        bed_level = self.bed_level
        
        # > Calculate water level + water depth
        water_level = interfaces.max(dim=dimn_interface)
        water_depth = (water_level - bed_level).rename(f'{gridname}_waterdepth')
        
        return water_depth
        
    @cached_property
    def bed_level(self):
        """Calculate bed level at each grid face.

        Computes bed elevation as the minimum interface elevation,
        averaged over time if available.

        Returns:
            xarray.DataArray: Bed elevation (positive downward) at each face.

        Raises:
            ValueError: If interfaces coordinate is not provided in data_description.
        """

        # Get dimensions
        dimn_interface = self.dimn_interface
        
        try:
            # > Get properties
            interfaces = self.interfaces
            # > Calculate bed level as minimum of interface height
            bed_level = interfaces.min(dim=dimn_interface).mean(dim='time')
        except:
            raise ValueError('For the bed level to be calculated, you need to provide the depth of the interface between the layers, specified as "interfaces". These can be varying in time.')
        return bed_level
        
    @cached_property
    def edge_length(self):
        """Calculate the length of each grid cell edge.

        Computes Euclidean distance between the two nodes of each edge
        in the horizontal (x,y) coordinate system.

        Returns:
            xarray.DataArray: Edge length at each edge.
        """
        
        # > Get properties
        edge_node_coords = self.edge_node_coords
        edge_length = calculate_distance_pythagoras(edge_node_coords[:,0,0], edge_node_coords[:,0,1], edge_node_coords[:,1,0], edge_node_coords[:,1,1]).rename(f'{self.gridname}_edge_length')

        return edge_length
    
    @cached_property
    def face_edge_weights(self):
        
        # > Get dimension names
        dimn_edges = self.dimn_edges
        fill_value = self.fill_value
        
        # > Get the edge-face connectivity
        face_edges = self.face_edges
        face_edges = face_edges.reset_coords(drop=True)
        face_coords = self.face_coords
        edge_coords = self.edge_coords
        
        # > Fill face-edge-connectivity matrix with face coordinates
        face_edge_coords = xr.where(face_edges!=fill_value, edge_coords.isel({dimn_edges: face_edges}), np.nan)
        
        # > Build the weights
        face_edge_weights = build_inverse_distance_weights(face_coords, face_edge_coords)
        
        return face_edge_weights
        
    @cached_property
    def edge_face_weights(self):

        # > Get dimension names
        dimn_faces = self.dimn_faces
        fill_value = self.fill_value
        gridname = self.gridname
        
        # > Get the edge-face connectivity
        edge_faces = self.edge_faces
        edge_faces = edge_faces.reset_coords(drop=True)
        
        face_coords = self.face_coords
        edge_coords = self.edge_coords

        # > Fill edge-face-connectivity matrix with face coordinates
        edge_face_coords = xr.where(edge_faces!=fill_value, face_coords.isel({dimn_faces:edge_faces}), np.nan)
        
        # > Build the weights
        edge_face_weights = build_inverse_distance_weights(edge_coords, edge_face_coords)
        
        # > Make into dask array
        edge_face_weights = edge_face_weights.reset_coords(drop=True)
        edge_face_weights = xr.DataArray(da.from_array(edge_face_weights), dims=[*edge_face_weights.dims], coords=edge_face_weights.coords, name='grid_edge_face_weights')
        
        return edge_face_weights

    def get_all_coordinates(self):

        uds = self.ds

        # > Get coordinate names
        coord_face_x, coord_face_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()
        coord_edge_x, coord_edge_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()
        coord_node_x, coord_node_y = uds.grid.to_dataset().mesh2d.attrs['node_coordinates'].split()

        # > Get dimension names
        dimn_faces = self.dimn_faces
        dimn_nodes = self.dimn_nodes
        dimn_edges = self.dimn_edges
        dimn_cart = self.dimn_cart

        # > Get face coordinates
        face_array = np.c_[uds.mesh2d_face_x, uds.mesh2d_face_y] # is NOT equal to uds.grid.face_coordinates
        face_coords = xr.DataArray(data=face_array, dims=[dimn_faces, dimn_cart], coords={f'{coord_face_x}':([dimn_faces], uds[f'{coord_face_x}']), f'{coord_face_y}':([dimn_faces], uds[f'{coord_face_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh face', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})

        # > Get edge coordaintes
        edge_array = uds.grid.edge_coordinates # np.c_[uds.mesh2d_edge_x, uds.mesh2d_edge_y]
        edge_coords = xr.DataArray(data=edge_array, dims=[dimn_edges, dimn_cart], coords={f'{coord_edge_x}':([dimn_edges], uds[f'{coord_edge_x}']), f'{coord_edge_y}':([dimn_edges], uds[f'{coord_edge_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh face', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})

        # > Get node coordinates
        node_array =  uds.grid.node_coordinates # np.c_[uds.mesh2d_node_x, uds.mesh2d_node_y]
        node_coords = xr.DataArray(data=node_array, dims=[dimn_nodes, dimn_cart], coords={f'{coord_node_x}':([dimn_nodes], uds[f'{coord_node_x}']), f'{coord_node_y}':([dimn_nodes], uds[f'{coord_node_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh node', 'bounds': 'mesh2d_node_x_bnd, mesh_node_y_bnd'})
    
        return face_coords, edge_coords, node_coords

    @cached_property
    def edge_node_coords(self):
            
        # > Get dimension names
        fill_value = self.fill_value
        dimn_nodes = self.dimn_nodes

        # > Get/buid edge-node connectivity
        edge_nodes = self.edge_nodes

        # > Build the node_coords
        # _, _, node_coords = self.get_all_coordinates()  # just get the node_coords
        node_coords = self.node_coords

        # > Get the coordinates of all nodes belonging to an edge
        edge_node_coords = xr.where(edge_nodes != fill_value, node_coords.isel({dimn_nodes: edge_nodes}), np.nan)

        return edge_node_coords 
    
    @cached_property
    def kzz(self, dicoww=5e-5, prandtl_schmidt=0.7):
        
        # > Get properties
        tracer = self.tracer.name
        gridname = self.gridname
        dimn_edges = self.dimn_edges
        viscosity = self.viscosity
        dicwwu = viscosity / prandtl_schmidt
        
        if tracer == f'{gridname}_sa1':
            k_l = (1/700) * 10e-6
        elif tracer == f'{gridname}_tem1': 
            k_l = (1/6.7) * 10e-6

        # > Calculate kzz
        kzz = dicwwu + dicoww + k_l
        
        # > Assign attributes and rename 
        kzz = kzz.assign_attrs({'mesh': f'{gridname}', 
                            'location': 'edge',
                            'cell_methods': f'{dimn_edges}: mean',
                            'standard_name': 'eddy_diffusivity',
                            'long_name': 'turbulent vertical eddy diffusivity', 
                            'units': 'm2 s-1', 
                            'grid_mapping': 'projected_coordinate_system'}).rename(f'{gridname}_dicwwu')
        
        # > Add calculated diffusivity to dataset
        self.ds[f'{gridname}_dicwwu'] = (kzz.dims, kzz.data)

        return kzz
    
    @cached_property
    def edge_faces(self):

        # > Get dimensions and fill value
        fill_value = self.fill_value
        dimn_edges = self.dimn_edges
        dimn_maxef = self.dimn_maxef
        
        edge_faces = xr.DataArray(self.ds.ugrid.grid.edge_face_connectivity, dims=(dimn_edges, dimn_maxef))

        return edge_faces

    @cached_property
    def edge_nodes(self):

        # > Get fill value, grid name and dimensions
        fill_value = self.fill_value
        dimn_edges = self.dimn_edges

        # > Get coordinate names
        coord_edge_x, coord_edge_y = self.ds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()

        # > Determine dimension name
        dimn_maxen = self.dimn_maxen

        # > Get connectivity
        edge_nodes = self.ds.grid.edge_node_connectivity

        # > Make into xr.DataArray with correct sizes, dimensions, and coordinates
        edge_nodes = xr.DataArray(data=edge_nodes, dims=[dimn_edges, dimn_maxen], coords={f'{coord_edge_x}':([dimn_edges], self.ds[f'{coord_edge_x}']), f'{coord_edge_y}':([dimn_edges], self.ds[f'{coord_edge_y}'])}, attrs={'cf_role': 'edge_node_connectivity', 'start_index':0, '_FillValue':fill_value}, name=self.ds.grid.to_dataset().mesh2d.attrs['edge_node_connectivity'])

        return edge_nodes
        
    @cached_property
    def face_edges(self):

        # > Get fill value, grid name, and dimensions
        fill_value = self.fill_value
        gridname = self.gridname
        dimn_faces = self.dimn_faces
        dimn_maxfn = self.dimn_maxfn

        # > Get voordinate names
        coord_face_x, coord_face_y = self.ds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()

        # > Get connectivity
        face_edges = self.ds.grid.face_edge_connectivity
        
        # > Make into xr.DataArray with correct sizes, dimensions, and coordinates
        face_edges = xr.DataArray(face_edges, dims=[dimn_faces, dimn_maxfn],
                                            coords={f'{coord_face_x}': ([dimn_faces], self.ds[f'{coord_face_x}']),
                                                    f'{coord_face_y}': ([dimn_faces], self.ds[f'{coord_face_y}'])},
                                            attrs={'cf_role': 'face_edge_connectivity', 'start_index': 0,
                                                    '_FillValue': fill_value}, name=f'{gridname}_face_edges')

        return face_edges
        
    @cached_property
    def unvs(self):
            
        from pyproj import Transformer

        # First check if the provided dataset is a xu.core.wrap.UgridDataset
        uds = self.ds

        # > Get dimensions, gridname, and coordinates
        gridname = uds.grid.name
        dimn_cart = self.dimn_cart

        varname_unvs = f'{gridname}_unvs'

        # > Get edge coordinate names
        edge_node_coords = self.edge_node_coords

        # > Check if the coordinate reference system is WGS84 (latitude, longitude)
        x1 = edge_node_coords[:, 0, 0]
        x2 = edge_node_coords[:, 1, 0]
        y1 = edge_node_coords[:, 0, 1]
        y2 = edge_node_coords[:, 1, 1]

        if uds.ugrid.crs[f'{gridname}'].name == 'WGS 84':

            # > Infer latitudes and longitudes from the edge-node-coordinates
            lat1 = y1
            lat2 = y2
            lon1 = x1
            lon2 = x2

            latlon2rd = Transformer.from_crs("epsg:4326",
                                     "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +towgs84=565.237,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812 +units=m +no_defs")
            # Transform coordinates
            x1_n, y1_n = latlon2rd.transform(lat1, lon1)
            x2_n, y2_n = latlon2rd.transform(lat2, lon2)

            # Give correct attiributes
            x1 = xr.DataArray(x1_n, dims=x1.dims, coords=x1.coords)
            x2 = xr.DataArray(x2_n, dims=x2.dims, coords=x2.coords)
            y1 = xr.DataArray(y1_n, dims=y1.dims, coords=y1.coords)
            y2 = xr.DataArray(y2_n, dims=y2.dims, coords=y2.coords)

        x = x2 - x1
        y = y2 - y1

        nf = nf = xr.concat([-y, x], dimn_cart).T #np.dstack([-y, x])
        vm_nf = nf.linalg.norm(dims=dimn_cart)

        # > Calculate the norm and divide by the norm
        unvs = nf / vm_nf 
        unvs = unvs.rename(varname_unvs)
        uds[f'{varname_unvs}'] = (unvs.dims, unvs.data)
                
        return unvs
    
#     def update_connectivities(self, boolean):
        
#         dimn_faces = self.dimn_faces
#         dimn_edges = self.dimn_edges
#         fill_value = self.fill_value
#         gridname = self.grid_name
        
#         if dimn_faces in boolean.dims:
            
#             # > Get voordinate names
#             coord_face_x, coord_face_y = self.ds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()
#             coord_edge_x, coord_edge_y = self.ds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'edge').split()  
            
#             # > Get dimensions
#             dimn_maxfn = self.dimn_maxfn
#             dimn_maxef = self.dimn_maxef
            
#             # > Start with chunking the boolean correctly
#             chunks = {dimn_faces:-1}
#             boolean = boolean.chunk(chunks)
            
#             # 1. The edge_faces parameter
#             # > Get edge_faces
#             edge_faces = self.edge_faces
            
#             # > Select the boolean on faces in the edge-face connectivity matrix
#             edge_faces_stacked = edge_faces.stack(__tmp_dim__=(dimn_edges, dimn_maxef))
#             boolean_ef_stacked = boolean.isel({dimn_faces: edge_faces_stacked})
#             boolean_ef = boolean_ef_stacked.unstack("__tmp_dim__")

#             # > Make into xr.DataArray with correct sizes, dimensions, and coordinates
#             # > Make the new face_edges 
#             edge_faces = xr.DataArray(edge_faces.where(boolean_ef, -999).values, dims=(dimn_edges, dimn_maxef))
            
#             # > Set the cached_property anew
#             self.edge_faces = edge_faces
            
#             # Get the boolean for whether the edges are valid (dim: meshd_nEdges)
#             nan_bool = (edge_faces == fill_value).all(dim=dimn_maxef)
            
#             # 2. The face_edges parameter
#             # > Get face_edges
#             face_edges = self.face_edges
            
#             # > Select the varname on faces in the edge-face connectivity matrix
#             face_edges_stacked = face_edges.stack(__tmp_dim__=(dimn_faces, dimn_maxfn))
#             boolean_fn_stacked = nan_bool.isel({dimn_edges: face_edges_stacked})
#             boolean_fn = boolean_fn_stacked.unstack("__tmp_dim__")
            
#             face_edges = face_edges.where(boolean_fn, -999)
            
#             face_edges = xr.DataArray(face_edges, dims=[dimn_faces, dimn_maxfn],
#                                             coords={f'{coord_face_x}': ([dimn_faces], self.ds[f'{coord_face_x}']),
#                                                     f'{coord_face_y}': ([dimn_faces], self.ds[f'{coord_face_y}'])},
#                                             attrs={'cf_role': 'face_edge_connectivity', 'start_index': 0,
#                                                     '_FillValue': fill_value}, name=f'{gridname}_face_edges')
            
#             self.face_edges = face_edges
            
#             return            
                 
    def uda_to_edges(self, uda):
        # > Define the to-be-interpolated tracer DataArray and grid
        varname = uda.name
        grid = self.ds.grid
        dimn_faces = self.dimn_faces
        
        if dimn_faces in uda.dims:

            # > Get dimension and grid names
            dimn_edges = self.dimn_edges
            fill_value = self.fill_value
            dimn_maxef = self.dimn_maxef

            # > Get the edge-face connectivity and replace fill values with -1
            edge_faces = self.edge_faces
            edge_faces = edge_faces.reset_coords(drop=True)
            edge_faces_validbool = (edge_faces!=fill_value).data
            edge_faces = edge_faces.where(edge_faces_validbool, -1)

            # > Make sure the face dimension is not chunked, otherwise we will 
            # > get "PerformanceWarning: Slicing with an out-of-order index is generating x times more chunks."
            chunks = {dimn_faces:-1}
            uda = uda.chunk(chunks)

            # > Select the varname on faces in the edge-face connectivity matrix
            edge_faces_stacked = edge_faces.stack(__tmp_dim__=(dimn_edges, dimn_maxef))
            edge_var_stacked = uda.isel({dimn_faces: edge_faces_stacked})
            edge_var = edge_var_stacked.unstack("__tmp_dim__")
            
            # > Convert data-array back to an xu.UgridDataArray
            edge_var = xu.UgridDataArray(edge_var, grid=grid)
            
            # > Set fill values to nan-values
            edge_var = edge_var.where(edge_faces_validbool, np.nan)

            # > Calculate the variable on the edges, based on the face_weights
            face_weights = self.edge_face_weights
            face_weights = face_weights.reset_coords(drop=True)

            # > Rechunk back with settin:g = "auto" #todo: provide option for custom chunksizes
            uda = uda.chunk(chunks="auto")
            edge_var = edge_var.chunk(chunks="auto")
            
            # If both of the edge vars are np.nan, then the sum should also be np.nan
            # Make a boolean for this to test the next step on
            nan_bool = ~edge_var.isnull().all(dim=dimn_maxef)
            
            # If one of the edges is nan, it's a boundary edge and this means that the corresponding
            # edge should be equal to the value in the non-nan cell > adjust the face_weights accordingly.
            # > First expand the face weights to correspond to the full edge_var array
            expanded_face_weights = face_weights.broadcast_like(edge_var)
            expanded_face_weights = expanded_face_weights.chunk("auto").chunk(edge_var.chunks)
            
            valid_mask = edge_var.notnull().reset_coords(drop=True)
            adjusted_face_weights = expanded_face_weights.where(valid_mask.data)
            
            # > Adjust the weights over the layer dimension so that the weights always add up to 1.
            adjusted_face_weights = (adjusted_face_weights / adjusted_face_weights.sum(dim=dimn_maxef, skipna=True))
            
            # Multipy edge variables with the face_weights to get the variable size on the edges
            edge_var = (edge_var * adjusted_face_weights).sum(dim=dimn_maxef)
            
            # Apply the boolean
            edge_var = edge_var.where(nan_bool)
            
            # edge_var = edge_var.mean(dim=dimn_maxef)
            # Give name and attributes
            attr_list = {'location': 'edge', 'cell_methods': f'{dimn_faces}: inverse distance weighted mean'}
            edge_var = edge_var.assign_attrs(attr_list).rename(varname)
            
            return edge_var
        
        else:
            raise ValueError(f'Variable {varname} does not contain dimension faces, so cannot be transformed from faces to edges.')
    
    def uda_to_faces(self, uda):
              
        # > Define the to-be-interpolated tracer DataArray and grid
        varname = uda.name
        grid = self.ds.grid
        
        dimn_nodes = self.dimn_nodes
        dimn_edges = self.dimn_edges
        
        if dimn_edges in uda.dims:
            
            dimn_faces = self.dimn_faces
            fill_value = self.fill_value
            dimn_maxfn = self.dimn_maxfn
            
            # > Get the edge-face connectivity and replace fill values with -1
            face_edges = self.face_edges
            face_edges = face_edges.reset_coords(drop=True)
            face_edges_validbool = (face_edges!=fill_value).data
            face_edges = face_edges.where(face_edges_validbool, -1)
            
            # > Make sure the face dimension is not chunked, otherwise we will 
            # > get "PerformanceWarning: Slicing with an out-of-order index is generating x times more chunks."
            chunks = {dimn_edges:-1}
            uda = uda.chunk(chunks)
            
            # > Select the varname on faces in the edge-face connectivity matrix
            face_edges_stacked = face_edges.stack(__tmp_dim__=(dimn_faces, dimn_maxfn))
            face_var_stacked = uda.isel({dimn_edges: face_edges_stacked})
            face_var = face_var_stacked.unstack("__tmp_dim__")
            
            # > Convert data-array back to an xu.UgridDataArray
            face_var = xu.UgridDataArray(face_var, grid=grid)
            
            # > Set fill values to nan-values
            face_var = face_var.where(face_edges_validbool, np.nan)
            
            # > Rechunk back with setting = "auto" #todo: provide option for custom chunksizes
            uda = uda.chunk(chunks="auto")
            face_var = face_var.chunk(chunks="auto")
            
            # If all of the edge vars are np.nan, then the sum should also be np.nan
            # Make a boolean for this to test the next step on
            nan_bool = ~face_var.isnull().all(dim=dimn_maxfn)
            
            # > Calculate the variable on the edges, based on the face_weights
            edge_weights = self.face_edge_weights
            
            # Multipy edge variables with the face_weights to get the variable size on the edges
            # face_var = face_var.mean(dim=dimn_maxfn)
            face_var = (face_var * edge_weights).sum(dim=dimn_maxfn)
            
            # # Apply the boolean
            face_var = face_var.where(nan_bool)
            
            # Update attributes from node/edge to face
            face_attrs = {'location': 'face', 'cell_methods': f'{dimn_faces}: mean'}
            face_var = face_var.assign_attrs(face_attrs)
            
            return face_var
    
        else:
            raise ValueError(f'Variable {varname} does not contain dimension edges, so cannot be transformed from edges to faces.')

    
    def compute_gradient_on_face(self, uda, scalar=False, add_to_dataset=False, mode_depth=None):

        # > Obtain uds and grid from constructorSVA object
        uds = self.ds
        grid = uds.grid

        # > Get dimension and grid names
        dimn_maxfn = self.dimn_maxfn
        dimn_faces = self.dimn_faces
        dimn_edges = self.dimn_edges
        fill_value = self.fill_value
        gridname = self.gridname
        dimn_cart = self.dimn_cart

        # > Get unit normal vectors
        unvs = self.unvs
        unvs = unvs.reset_coords(drop=True)
        
        # > Get the volume and flow area variables: check if they're in the 
        # > constructor first, then check the dataset, else throw error
        if scalar:
            flow_area = self.face_area
        else:
            flow_area = self.flow_area
        volume = self.volume

        if mode_depth == 'sum':
            # > Derive the summed cell volume
            volume = volume.sum(dim=self.dimn_layer, keep_attrs=True)
            # > Get the summed cell area
            flow_area = self.flow_area.sum(dim=self.dimn_layer, keep_attrs=True)
        elif mode_depth == 'mean':
            # > Get the average cell thickness
            cell_thickness = self.cell_thickness.mean(dim=self.dimn_layer, keep_attrs=True)
            # > Get the average cell volume from the average cell thickness
            volume = cell_thickness * self.cell_area
            # > Calculate the average flow area
            flow_area = flow_area.mean(dim=self.dimn_layer, keep_attrs=True)
        else:
            pass

        # > Get the face-edge connectivity and replace fill values with -1
        face_edges = self.face_edges
        # > Get boolean to mask other arrays too
        face_edges_validbool = (face_edges!=fill_value)
        # > Mask edges that don't have a value
        face_edges = face_edges.where(face_edges_validbool.data, -1)
        # > Stack face_edges for later use
        face_edges_stacked = face_edges.stack(__tmp_dim__=(dimn_faces, dimn_maxfn))
        
        # > Take the lentgth of the cartesian axis
        nc = unvs.sizes[dimn_cart]

        # Make (faces, maxfn) -> (faces, maxfn, cart)
#         face_edges_validbool3d = face_edges_validbool.expand_dims({dimn_cart: nc}).data
        
        # > Get the unit normal vectors (nf) also in the face-edges matrix
        fe_nfs = xr.where(face_edges_validbool, unvs.isel({dimn_edges: face_edges}), np.nan)
        fe_nfs = fe_nfs.chunk(chunks="auto")
        
        # // 2. See if the normal vectors are pointing out of the cell. If not, flip them.
        # > Calculate distance vectors
        dv = self.distance_vectors
        dv = dv.reset_coords(drop=True)
        
        # > Calculate the dot product between the calculated normal vectors and the distance vector for each face
        i_nfs = xr.dot(fe_nfs, dv, dims=[dimn_cart])
        # > if the product < 1, multiply by -1 to get an outwards facing normal vector, and update the variable
        fe_nfs = xr.where(i_nfs > 0, fe_nfs, fe_nfs * -1)

        # > Determine if we're looking at a velocity value u1 or u0
        # > Because these are vector quantities in the direction of the normal vector,
        # > to get to the final vector, we have to multiply u1/u0 by the normal vector
        # > first. Also, we need to check their sign for every edge.
        if not dimn_edges in uda.dims:
            uda = self.uda_to_edges(uda)
        else:
            pass

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

        # > Rechunk the data back to the original
        edge_var = edge_var.chunk(chunks="auto")
        uda = uda.chunk(chunks="auto")
        edge_au = edge_au.chunk(chunks="auto")

        # > Multiply the variable with the edge area (flow area), multiply by the 
        # > "flipped boolean" and the unit normal vector, and sum (dimension: faces)
        face_vars = (edge_var * edge_au * fe_nfs).sum(dim=dimn_maxfn, keep_attrs=True)
        
        # > Multiply the total result with (1/cell volume) (dimension: faces, cartesian_coordinates)
        gradient = (1/volume) * face_vars

        # > Check if it needs to be added to the dataset self.ds
        if add_to_dataset:
            # > Determine varname from the provided array
            varname = uda.name
            uds[f'{gridname}_{varname}_gradient'] = (gradient.dims, gradient.data)
            
        return gradient

    @cached_property
    def distance_vectors(self):

        # > Get dimensions, fill_value, and varname
        fill_value = self.fill_value
        dimn_edges = self.dimn_edges
        dimn_faces = self.dimn_faces
        dimn_cart = self.dimn_cart

        # Get  coordinate names
        coord_face_x, coord_face_y = self.ds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()

        # > Get face-edge-connectivity
        face_edges = self.face_edges
        face_edges = face_edges.reset_coords(drop=True)

        # > Get the cell face_coords!
        centroid_array = self.ds.grid.face_coordinates
        centroid_coords = xr.DataArray(data=centroid_array, dims=[dimn_faces, dimn_cart], coords={f'{coord_face_x}':([dimn_faces], self.ds[f'{coord_face_x}']), f'{coord_face_y}':([dimn_faces], self.ds[f'{coord_face_y}'])}, attrs={'units':'m', 'standard_name': 'projection_x_coordinate, projection_y_coordinate', 'long_name':'Characteristic coordinates of mesh centroids', 'bounds': 'mesh2d_face_x_bnd, mesh_face_y_bnd'})   

        # >> 1. Get the distance vector
        # > Get the edge coordinates
        edge_coords = self.edge_coords

        # > Put edge coordinates into face-edge connectivity matrix
        face_edge_coords = xr.where(face_edges!=fill_value, edge_coords.isel({dimn_edges: face_edges}), np.nan)

        # > Subtract the face coordinates from each of the edge coordinates in the connectivity matrix
        distance_vectors = face_edge_coords - centroid_coords

        return distance_vectors

    
    @cached_property
    def tracer_variance(self):
        # > Get properties
        tracer = self.tracer
        # > Calculate mean
        mean_tracer = tracer.mean(dim=self.dimn_layer)
        # > Calculate perturbation
        tracer_perturbation = tracer - mean_tracer
        tracer_variance = (tracer_perturbation**2).rename(f"{self.tracer.name}_variance")

        return tracer_variance 
    
    @cached_property
    def depth_integrated_tracer_variance(self):
        # > Get properties
        tracer_variance = self.tracer_variance
        depth = self.depth.drop_vars([n for n,v in self.depth.coords.items() if f"{self.dimn_layer}" in v.dims])
        depth_integrated_tracer_variance = integrate_trapz(tracer_variance, depth, dim=self.dimn_layer).rename(f"depth_integrated_{self.tracer.name}_variance")
        
        return depth_integrated_tracer_variance
     
    @cached_property
    def cell_thickness(self):

        cell_thickness = self.ds.mesh2d_flowelem_zw.diff(dim=self.dimn_interface).rename({f'{self.dimn_interface}':self.dimn_layer}).rename(f'{self.gridname}_cell_thickness')
        cell_thickness = cell_thickness.reset_coords(drop=True)

        return cell_thickness
    
    @cached_property
    def cell_area(self):
        # > Get the distance vectors first
        dv = self.distance_vectors
        dv = dv.reset_coords(drop=True)
        
        # > Get face_edges
        face_edges = self.face_edges
        face_edges = face_edges #.reset_coords(drop=True)
        # > Get boolean to mask other arrays too
        face_edges_validbool = (face_edges!=self.fill_value).data
        # > Mask edges that don't have a value
        face_edges = face_edges.where(face_edges_validbool, -1)
        # > Stack face_edges for later use
        face_edges_stacked = face_edges.stack(__tmp_dim__=(self.dimn_faces, self.dimn_maxfn))
        # > Get unit normal vectors
        unvs = self.unvs
        unvs = unvs #.reset_coords(drop=True)
        
        # > Get the unit normal vectors (nf) also in the face-edges matrix
        fe_nfs = xr.where(face_edges_validbool, unvs.isel({self.dimn_edges: face_edges}), np.nan)
        fe_nfs = fe_nfs.chunk(chunks="auto")
        
        # > Calculate the dot product between the calculated normal vectors and the distance vector for each face
        i_nfs = xr.dot(fe_nfs, dv, dims=[self.dimn_cart])
        # > if the product < 1, multiply by -1 to get an outwards facing normal vector, and update the variable
        fe_nfs = xr.where(i_nfs > 0, fe_nfs, fe_nfs * -1)
        
        # > Get the flow area
        flow_area = self.flow_area
        # > Fill face_edge matrix with flow area data
        edge_length = self.edge_length
        
        chunks = {dimn_edges:-1}
        edge_length = edge_length.chunk(chunks)
        edge_len_stacked = edge_length.isel({self.dimn_edges:face_edges_stacked})
        edge_len = edge_len_stacked.unstack("__tmp_dim__")
        
        # > Calculate the area
        cell_area = ((1/2)*(xr.dot(dv, fe_nfs, dim=[dimn_cart])*edge_len)).sum(dim=dimn_maxfn).rename(f'{self.gridname}_cell_area')
           
        # Incorrect as of 3-4-2025
        # cell_area = xr.DataArray(self.ds.grid.area, dims=(self.dimn_faces), name=f'{self.gridname}_cell_area')

        return cell_area
    
    @cached_property
    def volume(self):
        # > Get strings
        gridname = self.gridname
        
        # > Get properties
        cell_area = self.cell_area
        cell_thickness = self.cell_thickness
        
        volume = (cell_area * cell_thickness).rename(f'{self.gridname}_vol1')

        return volume
    
    def straining(self, integration='depth', depth_averaged=False):
        
        # > Get the tracer variance
        tracer_variance = self.tracer_variance

        # > Get the velocity perturbation
        ux_p = self.velx - self.velx.mean(dim=self.dimn_layer)
        uy_p = self.vely - self.vely.mean(dim=self.dimn_layer)

        # > Get the mean tracer over depth
        # tracer_mean = self.tracer.mean(dim=self.dimn_layer,  keep_attrs=True).rename(f'{self.gridname}_{self.tracer.name}a') # a stands for averaged

        # > Get the gradient of the tracer mean using the Leibniz rule determining that
        # > the mean of the gradient == the gradient of the mean
        tracer_gradient = self.compute_gradient_on_face(self.tracer, scalar=True) 
        tracer_mean_gradient = tracer_gradient.mean(dim=self.dimn_layer)
        # tracer_mean_gradient = self.compute_gradient_on_face(tracer_mean, depth_averaged=True)

        # > Calculate -2*sv2*ux_p and uy_p
        sv2_uxp = -2 * tracer_variance * ux_p
        sv2_uyp = -2 * tracer_variance * uy_p

        # > Calculate the dot product of the vector (-2*sv2*ux_p, -2*sv2*uy_p) with the tracer_mean_gradient
        dot_prod = tracer_mean_gradient.isel({f'{self.dimn_cart}':0})*sv2_uxp + tracer_mean_gradient.isel({f'{self.dimn_cart}':1})*sv2_uyp

        # > Drop coordinates that have coordinate to be integrated over before integrating so no difficulties arise
        dot_prod = dot_prod.drop_vars([n for n,v in dot_prod.coords.items() if f"{self.dimn_layer}" in v.dims])
        depth = self.depth.drop_vars([n for n,v in self.depth.coords.items() if f"{self.dimn_layer}" in v.dims])
                   
        if integration.lower() == 'depth':
            # > Integrate to get the straining term (trapezoidal rule)
            straining = integrate_trapz(dot_prod, depth, dim=self.dimn_layer)
        elif integration.lower() == 'volume':
            # > Multiply with volume per cell and sum over all cells
            straining = (dot_prod * self.volume).sum(dim=[f'{self.dimn_layer}', f'{self.dimn_faces}'])
        elif integration.lower() == 'none':
             straining = dot_prod
        else:
            raise ValueError(f"Could not derive what to do with integration={integration} keyword.")
        
        if depth_averaged:
            straining = straining * (1/self.water_depth)
            
        return straining.rename(f"{integration}_integrated_{self.tracer.name}_straining")
    
    def tendency(self, integration='depth', depth_averaged=False):
        
        # > Get tracer variance
        tracer_variance = self.tracer_variance

        # > Drop coordinates that have coordinate to be integrated over before integrating so no difficulties arise
        tracer_variance = tracer_variance.drop_vars([n for n,v in tracer_variance.coords.items() if f"{self.dimn_layer}" in v.dims])
        depth = self.depth.drop_vars([n for n,v in self.depth.coords.items() if f"{self.dimn_layer}" in v.dims])

        if integration.lower() == 'depth':
            # > Integrate tracer variance over depth (trapezoidal rule)
            tv_int = integrate_trapz(tracer_variance, depth, self.dimn_layer)
        elif integration.lower() == 'volume':
            # > Multiply with volume per cell and sum over all cells
            tv_int = (tracer_variance * self.volume).sum(dim=[f'{self.dimn_layer}', f'{self.dimn_faces}'])
        elif integration.lower() == 'none':
            tv_int = tracer_variance
        else:
            raise ValueError(f"Could not derive what to do with integration={integration} keyword.")
            
        tv_int = tv_int.unify_chunks()
        
        # > Take the difference in time (d/dt) 
        # > For this, chunk size in time must be larger than 1; check this first
        if any(val == 1 for val in tv_int.chunksizes['time']):
            tv_int = tv_int.chunk({'time': 4})
        else:
            pass

        tendency = tv_int.differentiate("time", datetime_unit="s").rename(f"{integration}_integrated_{self.tracer.name}_tendency")
        
        if depth_averaged:
            tendency = tendency * (1/self.water_depth)
            
        return tendency
       
    def advection(self, integration='depth',  depth_averaged=False):

        # > First calculate (S')^2 * u and (S')^2 * v
        u_sv2 = self.velx * self.tracer_variance
        v_sv2 = self.vely * self.tracer_variance

        # > Drop coordinates that have coordinate to be integrated over before integrating so no difficulties arise
        u_sv2 = u_sv2.drop_vars([n for n,v in u_sv2.coords.items() if f"{self.dimn_layer}" in v.dims])
        v_sv2 = v_sv2.drop_vars([n for n,v in v_sv2.coords.items() if f"{self.dimn_layer}" in v.dims])
        depth = self.depth.drop_vars([n for n,v in self.depth.coords.items() if f"{self.dimn_layer}" in v.dims])

        # > Integrate terms in x and y direction
        # u_sv2_int = integrate_trapz(u_sv2, depth, dim=self.dimn_layer).rename(f"{self.gridname}_{self.tracer.name}_adv_x")
        # v_sv2_int = integrate_trapz(v_sv2, depth, dim=self.dimn_layer).rename(f"{self.gridname}_{self.tracer.name}_adv_y")

        # > Calculate gradient
        grad_usv2 = self.compute_gradient_on_face(u_sv2) 
        grad_vsv2 =  self.compute_gradient_on_face(v_sv2)

        # > Cartesian dimension 0 is x-direction, 1 is y-direction
        # > We need du/dx + dv/dy for the horizontal divergence of the velocity * salinity variance vector
        adv = grad_usv2.isel({f'{self.dimn_cart}':0}) + grad_vsv2.isel({f'{self.dimn_cart}':1})
        
        if integration.lower() == 'depth':
            # > Integrate advection over depth (trapezoidal rule)
            # > Applying the Leibniz rule, we know that the divergence of an integral with fixed limits
            # > is equal to the integral of the divergence, so we take the integral of the previously calculated advection term
            advection = integrate_trapz(adv, depth, dim=self.dimn_layer)
        elif integration.lower() == 'volume':
            # > Multiply with volume per cell and sum over all cells
            advection = (adv * self.volume).sum(dim=[f'{self.dimn_layer}', f'{self.dimn_faces}'])
        elif integration.lower() == 'none':
            advection = adv
        else:
            raise ValueError(f"Could not derive what to do with integration={integration} keyword.")
            
        if depth_averaged:
            advection = advection * (1/self.water_depth)
            
        return advection.rename(f"{integration}_integrated_{self.tracer.name}_advection")
    
    def horizontal_dissipation(self, integration='depth', depth_averaged=False):
        # > Get the dimensions
        dimn_cart = self.dimn_cart
        # > Get the horizontal diffusion
        diu = self.diu
        
        # > Check if the horizontal diffusivity is defined on the edges, and if it is,
        # > interpolate it on the faces
        if self.dimn_edges in diu.dims:
            diu = self.uda_to_faces(diu)
            ho
        try:
            tracer_grad = self.compute_gradient_on_face(self.tracer, scalar=True)
            dsdx = tracer_grad.isel({dimn_cart:0}).transpose(*list(diu.dims)).reset_coords(drop=True)
            dsdy = tracer_grad.isel({dimn_cart:1}).transpose(*list(diu.dims)).reset_coords(drop=True)

            hor_dis = 2*diu*(dsdx**2 + dsdy**2)
            
            # > Drop coordinates that have coordinate to be integrated over before integrating so no difficulties arise
            hor_dis = hor_dis.drop_vars([n for n,v in hor_dis.coords.items() if f"{self.dimn_layer}" in v.dims])
            depth = self.depth.drop_vars([n for n,v in self.depth.coords.items() if f"{self.dimn_layer}" in v.dims])

            # > Integrate the total term and multiply with -1 
            if integration.lower() == 'depth':
                # > Integrate dissipation over depth (trapezoidal rule)
                dissipation = integrate_trapz(hor_dis, depth, self.dimn_layer)
            elif integration.lower() == 'volume':
                # > Multiply with volume per cell and sum over all cells
                dissipation = ((hor_dis * self.volume).sum(dim=[f'{self.dimn_layer}', f'{self.dimn_faces}']))
            elif integration.lower() == 'none':
                dissipation = hor_dis
            else:
                raise ValueError(f"Could not derive what to do with integration={integration} keyword.")

            if depth_averaged:
                dissipation = dissipation * (1/self.water_depth)
        
            return dissipation.rename(f"{integration}_integrated_{self.tracer.name}_horizontal_dissipation")
        
        except:
            return None
        
    def dissipation(self, integration='depth', depth_averaged=False):
        # > Get the vertical turbulent diffusivity
        kzz = self.kzz
        
        # > See if the depth dimension is in the kzz variable, if not then it's a staggered grid.
        # > If staggered grid, then interpolate to dimension
        if (self.dimn_interface != None) and (self.dimn_interface in kzz.dims):
            kzz = dfmt.uda_interfaces_to_centers(kzz)

        # > Check if the vertical eddy diffusivity is defined on the edges, and if it is,
        # > interpolate it on the efaces
        if self.dimn_edges in kzz.dims:
            kzz = self.uda_to_faces(kzz)

        # > Differentiate the tracer over depth
        dsdz = differentiate_over_3d_coord(self.tracer, f"{self.depth.name}", axis=-1)
        dsdz_sq = 2 * kzz * (dsdz**2)

        # > Drop coordinates that have coordinate to be integrated over before integrating so no difficulties arise
        dsdz_sq = dsdz_sq.drop_vars([n for n,v in dsdz_sq.coords.items() if f"{self.dimn_layer}" in v.dims])
        depth = self.depth.drop_vars([n for n,v in self.depth.coords.items() if f"{self.dimn_layer}" in v.dims])
            
        # > Integrate the total term and multiply with -1 
        if integration.lower() == 'depth':
            # > Integrate dissipation over depth (trapezoidal rule)
            dissipation = integrate_trapz(dsdz_sq, depth, self.dimn_layer)
        elif integration.lower() == 'volume':
            # > Multiply with volume per cell and sum over all cells
            dissipation = ((dsdz_sq * self.volume).sum(dim=[f'{self.dimn_layer}', f'{self.dimn_faces}']))
        elif integration.lower() == 'none':
            dissipation = dsdz_sq
        else:
            raise ValueError(f"Could not derive what to do with integration={integration} keyword.")
        
        if depth_averaged:
            dissipation = dissipation * (1/self.water_depth)
            
        return dissipation.rename(f"{integration}_integrated_{self.tracer.name}_dissipation")
               
    
    # def _setup(self, data):
    #     """Iterates over keys in dictionary. Handles 4d-data, if one argument is left empty, dummy dimension will be created.
    #     Args:
    #         data (dict): Dictionary that describes geospatial dimensions of the dataset.
    #     """
    #     for dimension in data.keys():
    #         if data[dimension] is None:
    #             self.ds = self.ds.expand_dims(dimension)
    #         else:
    #             self.ds = self.ds.rename({data[dimension]: dimension})

    #     self.ds = self.ds.transpose("time",
    #                                 "depth",
    #                                 "nfaces",
    #                                 "nedges",
    #                                 "nnodes",
    #                                 ...)
        
        # removed the nfaces and nedges dimnesion, as they are derived by the xugrid package
