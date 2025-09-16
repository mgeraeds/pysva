# pySVA

The pySVA toolbox is Python-based open-source software for the calculation of terms in a salinity variance budget that can be applied to model output from models with unstructured, staggered C-grid meshes such as [D-FLOW Flexible Mesh](https://content.oss.deltares.nl/delft3d/D-Flow_FM_User_Manual.pdf). 

## The salinity variance budget
Burchard et al. (2008) first described the total salinity variance equation. %to link physical and numerical mixing using an idealised model. 
The premise of their work was that tracer variance decay can be used to discern both physical and numerical mixing, which they tested on an idealised model. 
To derive the salinity variance equation, Burchard et al. (2008) started from the Reynolds-averaged salt conservation advection-diffusion equation in a three-dimensional domain (Burchard et al. 2008, Li et al. 2018),

![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20t%7D&plus;%5Cmathbf%7Bu%7D%5Ccdot%5Cnabla%20S-%5Cnabla%5Ccdot(%5Cmathbf%7BK%7D%5Cnabla%20S)=0,)

where ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5Cmathbf%7Bu%7D=(u,v,w))is the three-dimensional velocity vector and ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5Cmathbf%7BK%7D=(K_%7Bxx%7D,K_%7Byy%7D,K_%7Bzz%7D)) is the diffusivity tensor. Applying a Reynolds decomposition to the total salinity and velocity vector, these can be decomposed in a volume mean and varying part as ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7DS=%5B%5B%7BS%7D%5D%5D&plus;S'_%7Btot%7D) and ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5Cmathbf%7Bu%7D=%5Cmathbf%7B%5Cbar%7Bu%7D%7D&plus;%5Cmathbf%7Bu%7D') respectively, with ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5B%5B%5Ccdot%5D%5D) denoting the volume average.  
Substituting this decomposition into the salt conservation equation, Burchard et al. (2008) obtained an expression for the conservation of volume mean salinity. Taking the difference between the salt conservation equation and the obtained expression for conservation of volume mean salinity and multiplying this difference with $2S'$ yields the salinity variance equation:

![equation](https://latex.codecogs.com/svg.image?%5Cbg%7Bwhite%7D%5Cfrac%7B%5Cpartial(S'_%7Btot%7D)%5E2%7D%7B%5Cpartial%20t%7D&plus;%5Cnabla%5Ccdot%5B%5Cmathbf%7Bu%7D(S'_%7Btot%7D)%5E2-%5Cmathbf%7BK%7D%5Cnabla(S'_%7Btot%7D)%5E2%5D-2S'%5B%5B%5Cmathbf%7Bu%7D'%5Ccdot%5Cnabla%20S'_%7Btot%7D%5D%5D=-2(%5Cmathbf%7BK%7D%5Cnabla%20S'_%7Btot%7D)%5Ccdot(%5Cnabla%20S'_%7Btot%7D).)

In a paper applying the principle of salinity variance to the Changjiang estuary, \cite{li_transformation_2018} built on this method by decomposing the total salinity variance into horizontal and vertical contributions. The individual horizontal and vertical contributions can be calculated by decomposing the total volume-integrated salinity variance. Since it holds that ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7DS=%5B%5BS%5D%5D&plus;S'_%7Btot%7D), with ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5B%5BS%5D%5D) the volume-averaged salinity and ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7DS'_%7Btot%7D) the deviation from the volume average, and it also holds that ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7DS=%5Coverline%7BS%7D&plus;S'_v), where ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5Coverline%7BS%7D) is the depth-averaged salinity and ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7DS'_v) is the vertical deviation from the depth-averaged salinity, the following relations can be derived:

![equation](https://latex.codecogs.com/svg.image?%5Cbg%7Bwhite%7D%5Cbegin%7Balign%7D(S'_v)%5E2&=(S-%5Coverline%7BS%7D)%5E2%5C%5C(S'_h)%5E2&=(%5Coverline%7BS%7D-%5B%5BS%5D%5D)%5E2,%5Cmathrm%7Band%7D%5C%5C(S'_%7Btot%7D)%5E2&=(S-%5B%5BS%5D%5D)%5E2,%5C%5C%5Cend%7Balign%7D)

where ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D(S'_h)%5E2) is the horizontal salinity variance, ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D(S'_v)%5E2) is the vertical salinity variance, and ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D(S'_{tot})%5E2) is the total salinity variance. 

The vertical salinity variance equation can be derived using these variance relations and the salinity variance balance of Burchard et al. (2008). 
Considering a single vertical water column and decomposing the salinity and velocity in a depth-averaged mean and a perturbation from the vertical average, where ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7DS=%5Coverline%7BS%7D&plus;S'_v) and ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5Cmathbf%7Bu%7D=%5Cmathbf%7B%5Coverline%7Bu%7D%7D&plus;%5Cmathbf%7Bu'_v%7D), Li et al. (2018) obtained the following expression:
![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5Cunderbrace%7B%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20t%7D%5Cint(S'_v)%5E2%5Cmathrm%7Bd%7Dz%7D_%7B%5Cmathrm%7Btendency%7D%7D=-%5Cunderbrace%7B%5Cnabla_h%5Ccdot%5Cint%5Cmathbf%7Bu_h%7D(S'_v)%5E2%5Cmathrm%7Bd%7Dz%7D_%7B%5Cmathrm%7Badvection%7D%7D&plus;%5Cunderbrace%7B%5Cint-2%5Cmathbf%7Bu'_v%7DS'_v%5Ccdot%5Cnabla%5Coverline%7BS%7D%5Cmathrm%7Bd%7Dz%7D_%7B%5Cmathrm%7Bstraining%7D%7D-%5Cunderbrace%7B%5Cint%202%5Cfrac%7B%5Cpartial%5E2%20K%7D%7B%5Cpartial%20z%5E2%7D%5Cbigg(%5Cfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20z%7D%5Cbigg)%5E2%5Cmathrm%7Bd%7Dz%7D_%7B%5Cmathrm%7Bdissipation%7D%7D-%5Cunderbrace%7B%5Cint%5Cmathcal%7BM%7D_%7B%5Cmathrm%7Bnum%7D%7D%5Cmathrm%7Bd%7Dz%7D_%7B%5Cmathrm%7Bnum.mixing%7D%7D,)

The left-hand side of this equation represents the temporal change of the salinity variance in time, referred to as the _tendency_. The first three terms on the right-hand side _advection, straining_ and _dissipation_, respectively---can be used to quantify and visualise the mechanisms responsible for the spatial patterns of stratification within a coast-delta system and thus give us insight into the system's behaviour. This package can be used to calculate these four different terms based on the provided model data.

## Installation
This package is still under development, which is why it is most convenient to install the most recent release from GitHub directly. Use git to clone the repository using:
```
$ git clone https://github.com/mgeraeds/pysva.git
```
Navigate to the directory where your cloned repository lives. You can find the _setup.py_ file in this directory. Install the package in development mode using:
```
$ pip install .
```

## Logic structure and usage
The core of the package is built around a constructor class (\texttt{constructorSVA}) that initialises the analysis environment from either a file path or an in-memory dataset. It serves as the entry point for hydrodynamic and tracer analyses. This design ensures flexibility in handling different data sources, such as \texttt{xarray.Dataset}, \texttt{xarray.DataArray}, \texttt{xugrid.UgridDataset}, \texttt{xugrid.UgridDataArray}, or other UGRID-compliant data structures.

The constructor accepts as input either a NetCDF file or an \href{https://docs.xarray.dev/en/stable/}{xarray}-based object, together with a dictionary describing the relevant physical variables (e.g., velocity components, tracer, volume, flow area). Upon initialisation, the class reads the dataset. It automatically attaches references to fundamental hydrodynamic variables (velocity components, volume, viscosity, tracer concentration, depth), as well as optional fields such as layer interfaces. Grid-related attributes (node/edge/face coordinates, connectivity, and dimensional metadata) are stored for subsequent computations. This design abstracts away dataset-specific details and provides a standardised interface for subsequent diagnostics and numerical experiments.

               ┌──────────────────────────────┐
               │           Inputs             │
               │ ──────────────────────────── │
               │ • NetCDF file (path)         │
               │ • xarray.Dataset             │
               │ • xarray.DataArray           │
               │ • xu.UgridDataset            │
               │ • data_description (dict)    │
               └───────────────┬──────────────┘
                               │
                               ▼
               ┌──────────────────────────────┐
               │          Constructor         │
               │  ─────────────────────────── │
               │ • Reads dataset              │
               │ • Extracts grid + dimensions │
               │ • Assigns core variables     │
               │   (velx, vely, velz, tracer, │
               │    viscosity, depth, etc.)   │
               │ • Sets optional vars         │
               │   (volume, flow_area, etc.)  │
               └───────────────┬──────────────┘
                               │
                               ▼
          ┌────────────────────────────────────────┐
          │      Cached geometry properties        │
          │ ────────────────────────────────────── │
          │ • flow_area (Algorithm 5, TRM eqns)    │
          │ • face_area                            │
          │ • water_depth                          │
          │ • bed_level                            │
          │ • edge_length (pythagoras distance)    │
          │ • face_edge_weights, edge_face_weights │
          │ • face_coords, edge_coords, node_coords│
          │ • edge_node_coords                     │
          └───────────────────┬────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │       Outputs / API          │
               │ ──────────────────────────── │
               │ • Standardised grid metadata │
               │ • Hydrodynamic variables     │
               │ • Derived diagnostics        │
               │   (kzz, tracer variance)     │
               └──────────────────────────────┘


A more detailed explanation on the usage of the package can be found in the documentation. Examples are included in the notebooks folder.

## License
This software is licensed under an MIT license. Details on the license can be found in the LICENSE file.

## Project status
🚨🚨 The development of this package has been stalled. Issues can still be opened, but fixes might take longer.

## References
[1] Hans Burchard, Hannes Rennau, _Comparative quantification of physically and numerically induced mixing in ocean models_, Ocean Modelling, 20(3), 2008, Pages 293-311,ISSN 1463-5003, doi:10.1016/j.ocemod.2007.10.003.

[2] Li, X., Geyer, W. R., Zhu, J., & Wu, H. (2018). The Transformation of Salinity Variance: A New Approach to Quantifying the Influence of Straining and Mixing on Estuarine Stratification. Journal of Physical Oceanography, 48(3), 607-623. https://doi-org.tudelft.idm.oclc.org/10.1175/JPO-D-17-0189.1
