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
        self.face_nodes = self.face_nodes
        self.kzz = self.kzz
        self.tracer_variance = None
        self._tracer_perturbation = None
        self._velocity_perturbation = None

    def _read(self, file_name, **kwargs):
        # self.ds = xr.open_dataset(file_name, use_cftime=True, **kwargs)
        self.ds = dfmt.open_partitioned_dataset(file_name, **kwargs)

    @property
    def edge_face_weights(self):
        if not hasattr(self, 'edge_face_weights'):
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

        if not hasattr(self, 'face_coords'):
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
        
        return face_coords, edge_coords, node_coords

    @property
    def edge_node_coords(self):
        if not hasattr(self, 'edge_node_coords'):
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
    
    @property
    def kzz(self, dicoww=5e-5, prandtl_schmidt=0.7):
        if not hasattr(self, 'kzz'):
            tracer = self.tracer
            uds = self.ds
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
            uds[f'{gridname}_dicwwu'] = (kzz.dims, kzz.data)

        return kzz
    
    @property
    def edge_faces(self):
        if not hasattr(self, 'edge_faces'):
            # > Get dimensions and fill value
            fill_value = self.fill_value
            dimn_edges = self.dimn_edges
            dimn_maxef = self.dimn_maxef
            
            edge_faces = xr.DataArray(self.ds.ugrid.grid.edge_face_connectivity, dims=(dimn_edges, dimn_maxef))
            edge_faces_validbool = edge_faces!=fill_value
            edge_faces = edge_faces.where(edge_faces_validbool, -1)

        return edge_faces

    @property
    def edge_nodes(self):
        if not hasattr(self, 'edge_nodes'):
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
        
    @property
    def face_edges(self):
        if not hasattr(self, 'face_edges'):
            # > Get fill value, grid name, and dimensions
            fill_value = self.fill_value
            gridname = self.gridname
            dimn_faces = self.dimn_faces
            dimn_maxfn = self.dimn_maxfn

            # > Get voordinate names
            coord_face_x, coord_face_y = self.ds.grid.to_dataset().mesh2d.attrs['node_coordinates'].replace('node', 'face').split()

            # > Get connectivity
            face_edges = self.face_edges

            # > Make into xr.DataArray with correct sizes, dimensions, and coordinates
            face_edge_connectivity = xr.DataArray(face_edges, dims=[dimn_faces, dimn_maxfn],
                                                coords={f'{coord_face_x}': ([dimn_faces], self.ds[f'{coord_face_x}']),
                                                        f'{coord_face_y}': ([dimn_faces], self.ds[f'{coord_face_y}'])},
                                                attrs={'cf_role': 'face_edge_connectivity', 'start_index': 0,
                                                        '_FillValue': fill_value}, name=f'{gridname}_face_edges')
        return face_edge_connectivity
        
    @property
    def unvs(self):
        if not hasattr(self, 'unvs'):
            import pyproj

            # First check if the provided dataset is a xu.core.wrap.UgridDataset
            uds = constructorSVA.ds

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

                # > Define coordinate reference system
                geodesic = pyproj.Geod(ellps='WGS84')

                # > Infer latitudes and longitudes from the edge-node-coordinates
                lat1 = y1
                lat2 = y2
                lon1 = x1
                lon2 = x2
                # > Calculate distance vector
                fwd_azimuth, _, distance = geodesic.inv(lat2, lon2, lat1, lon1)
                az_rad = np.deg2rad(fwd_azimuth)
                x = np.sin(az_rad) * distance
                y = np.cos(az_rad) * distance

            else:
                x = x2 - x1
                y = y2 - y1

            nf = nf = xr.concat([-y, x], dimn_cart).T #np.dstack([-y, x])
            vm_nf = nf.linalg.norm(dims=dimn_cart)

            # > Calculate the norm and divide by the norm
            unvs = nf / vm_nf 
            unvs = unvs.rename(varname_unvs)
            uds[f'{varname_unvs}'] = (unvs.dims, unvs.data)
                
        return unvs

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
