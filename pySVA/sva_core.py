__all__ = ['constructorSVA']

# tef.tracer = tef.ds.salt
# data_description = {
#     "depth" : "mesh2d_nLayers",
#     "time" : "time",
#     "faces" : "mesh2d_nFaces"
# }

# Do imports
import xarray as xr
import xugrid as xu
import dfm_tools as dfmt
import numpy as np
from functools import cached_property
from pySVA.sva_helpers import build_inverse_distance_weights

class constructorSVA:
    def __init__(self, input_file, data_description, **kwargs): # removed data description data_description,

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
        # else:
            raise IOError("Please provide xr.Dataset, xu.core.wrap.UgridDataset, xr.DataArray, or a file path to a netCDF.")

        # self._setup(data=data_description)
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

        self.face_coords, self.edge_coords, self.node_coords = self.get_all_coordinates()

        # Attributes
        self.velx = self.ds[f'{data_description["velx"]}']
        self.vely = self.ds[f'{data_description["vely"]}']
        self.velz = self.ds[f'{data_description["velz"]}']
        self.flow_area = self.ds[f'{data_description["flow_area"]}']
        self.volume = self.ds[f'{data_description["volume"]}']
        self.viscosity = self.ds[f'{data_description["viscosity"]}']
        self.tracer = self.ds[f'{data_description["tracer"]}']

        # Hidden/calculated values
        self.edge_nodes = self.edge_nodes
        self.face_edges = self.face_edges
        self.kzz = self.kzz
        self.tracer_variance = self.tracer_variance
        # self._tracer_perturbation = None
        # self._velocity_perturbation = None

    def _read(self, file_name, **kwargs):
        # self.ds = xr.open_dataset(file_name, use_cftime=True, **kwargs)
        self.ds = dfmt.open_partitioned_dataset(file_name, **kwargs)

    @cached_property
    def edge_face_weights(self):

        # > Get dimension names
        dimn_faces = self.dimn_faces
        fill_value = self.fill_value
        
        # > Get the edge-face connectivity
        edge_faces = self.edge_faces
        
        face_coords = self.face_coords
        edge_coords = self.edge_coords

        # > Fill edge-face-connectivity matrix with face coordinates
        edge_face_coords = xr.where(edge_faces!=fill_value, face_coords.isel({dimn_faces:edge_faces}), np.nan)
        
        # > Build the weights
        edge_face_weights = build_inverse_distance_weights(edge_coords, edge_face_coords)
        
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
    
    def uda_to_edges(self, uda):
        # > Define the to-be-interpolated tracer DataArray and grid
        varname = uda.name
        grid = self.ds.grid
        dimn_faces = self.dimn_faces

        if dimn_faces in uda.dims:

            # > Get dimension and grid names
            dimn_faces = self.dimn_faces
            dimn_edges = self.dimn_edges
            fill_value = self.fill_value
            dimn_maxef = self.dimn_maxef

            # > Get the edge-face connectivity and replace fill values with -1
            edge_faces = self.edge_faces
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
            face_weights = self.edge_face_weights
            edge_var = (edge_var * face_weights).sum(dim=dimn_maxef)

            # edge_var = edge_var.mean(dim=dimn_maxef)
            # Give name and attributes
            attr_list = {'location': 'edge', 'cell_methods': f'{dimn_faces}: inverse distance weighted mean'}
            edge_var = edge_var.assign_attrs(
                    attr_list).rename(varname)
            
            return edge_var
        
        else:
            raise ValueError(f'Variable {varname} does not contain dimension faces, so cannot be transformed from faces to edges.')

    def compute_gradient_on_face(self, uda, add_to_dataset=False):

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
        
        # > Get the volume and flow area variables: check if they're in the 
        # > constructor first, then check the dataset, else throw error
        flow_area = self.flow_area
        volume = self.volume
            
        # > Get the face-edge connectivity and replace fill values with -1
        face_edges = self.face_edges
        # > Get boolean to mask other arrays too
        face_edges_validbool = face_edges!=fill_value
        # > Mask edges that don't have a value
        face_edges = face_edges.where(face_edges_validbool, -1)

        # > Get the unit normal vectors (nf) also in the face-edges matrix
        fe_nfs = xr.where(face_edges_validbool, unvs.isel({dimn_edges: face_edges}), np.nan)

        # // 2. See if the normal vectors are pointing out of the cell. If not, flip them.
        # > Calculate distance vectors
        dv = self.distance_vectors
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

        # > Stack face_edges for later use
        face_edges_stacked = face_edges.stack(__tmp_dim__=(dimn_faces, dimn_maxfn))

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

        # > Get the cell centroid coordinates
        # > NOTE: this is different from the face_coords!
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

        mean_tracer = self.tracer.mean(dim=self.dimn_layer)
        tracer_perturbation = self.tracer - mean_tracer
        tracer_variance = tracer_perturbation**2

        return tracer_variance 
    
    @cached_property
    def advection(self):

        # > First calculate (S')^2 * u and (S')^2 * v
        u_sv2 = self.velx * self.tracer_variance
        v_sv2 = self.vely * self.tracer_variance

        # > Integrate terms in x and y direction
        u_sv2_int = u_sv2.integrate(self.dimn_layer).rename(f"{self.gridname}_horizontal_tracer_advection")
        v_sv2_int = v_sv2.integrate(self.dimn_layer).rename(f"{self.gridname}_vertical_tracer_advection")

        # > Interpolate values to edges
        u_sv2_int = self.uda_to_edges(u_sv2_int)
        v_sv2_int = self.uda_to_edges(v_sv2_int)

        # > Calculate gradient
        grad_usv2 = self.compute_gradient_on_face(u_sv2_int) 
        grad_vsv2 =  self.compute_gradient_on_face(v_sv2_int)

        # > Cartesian dimension 0 is x-direction, 1 is y-direction
        # > We need du/dx + dv/dy for the horizontal divergence of the velocity * salinity variance vector
        
        
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
