# pySVA

The pySVA toolbox is Python-based open-source software for the calculation of terms in a salinity variance budget that can be applied to model output from models with unstructured, staggered C-grid meshes such as [D-FLOW Flexible Mesh](https://content.oss.deltares.nl/delft3d/D-Flow_FM_User_Manual.pdf). 

## The salinity variance budget
Burchard et al. (2008) first described the total salinity variance equation. %to link physical and numerical mixing using an idealised model. 
The premise of their work was that tracer variance decay can be used to discern both physical and numerical mixing, which they tested on an idealised model. 
To derive the salinity variance equation, Burchard et al. (2008) started from the Reynolds-averaged salt conservation advection-diffusion equation in a three-dimensional domain (Burchard et al. 2008, Li et al. 2018),

$$
\begin{equation}
\frac{\partial S}{\partial t} + \mathbf{u} \cdot \nabla S -\nabla \cdot (\mathbf{K} \nabla S) = 0,
\end{equation}
$$

where $`\mathbf{u} = (u, v, w)`$ is the three-dimensional velocity vector and $\mathbf{K} = (K_{xx} + K_{yy} + K_{zz})$ is the diffusivity tensor. Applying a Reynolds decomposition to the total salinity and velocity vector, these can be decomposed in a volume mean and varying part as $`S=[[S]] + S'_{tot}`$ and  $\mathbf{u} = [[\mathbf{u}]] + \mathbf{u'}$ respectively, with $[[\cdot]]$ denoting the volume average.  

Substituting this decomposition into the salt conservation equation, Burchard et al. (2008) obtained an expression for the conservation of volume mean salinity. Taking the difference between the salt conservation equation and the obtained expression for conservation of volume mean salinity and multiplying this difference with $2S'$ yields the salinity variance equation:

$$
\begin{equation}
\frac{\partial(S'_{tot})^2}{\partial t}+ \nabla\cdot[\mathbf{u}(S'_{tot})^2-\mathbf{K}\nabla(S'_{tot})^2]-2S'[[\mathbf{u}'\cdot\nabla S'_{tot}]] = -2(\mathbf{K}\nabla S'_{tot})\cdot (\nabla S'_{tot}).
\end{equation}
$$

In a paper applying the principle of salinity variance to the Changjiang estuary, Li et al. (2018) built on this method by decomposing the total salinity variance into horizontal and vertical contributions. The individual horizontal and vertical contributions can be calculated by decomposing the total volume-integrated salinity variance. Since it holds that $`S=[[S]] + S'_{tot}`$, with $[[S]]$ the volume-averaged salinity and $S'_{tot}$ the deviation from the volume average, and it also holds that $S = \overline{S} + S'_v$, where $\overline{S}$ is the depth-averaged salinity and  $S'_v$ is the vertical deviation from the depth-averaged salinity, the following relations can be derived:

$$
\begin{align}
    (S'_v)^2 &= (S-\overline{S})^2\\
    (S'_h)^2 &= (\overline{S} - [[S]])^2, \textrm{~and,}\\
    (S'_{tot})^2 &= (S-[[S]])^2,
\end{align}
$$

where $(S_h')^2$ is the horizontal salinity variance, $(S_v')^2$ is the vertical salinity variance, and $(S'_{tot})^2$ is the total salinity variance.

The vertical salinity variance equation can be derived using these variance relations and the salinity variance balance of Burchard et al. (2008). 
Considering a single vertical water column and decomposing the salinity and velocity in a depth-averaged mean and a perturbation from the vertical average, where $S=\overline{S} + S'_v$ and $\mathbf{u} = \mathbf{\overline{u}}+\mathbf{u'_v}$, Li et al. (2018) obtained the following expression:


$$
\begin{equation}
\begin{split}
\underbrace{\frac{\partial}{\partial t}\int(S'_v)^2 \mathrm{d}z}_{\textrm{tendency}}~=~-&\underbrace{\nabla_h\cdot\int \mathbf{u_h}(S'_v)^2 \mathrm{d} z}_{\textrm{advection}} + \underbrace{\int-2\mathbf{u'_v}S'_v\cdot\nabla\overline{S}\mathrm{d} z }_{\textrm{straining}} \\
&- \underbrace{ \int 2\frac{\partial^2 K}{\partial z^2}\bigg(\frac{\partial S}{\partial z}\bigg)^2 \mathrm{d} z }_{\textrm{dissipation}} -\underbrace{\int\mathcal{M}_{\mathrm{num}}\mathrm{d} z}_{\textrm{num. mixing}},
\end{split}
\end{equation}
$$

The left-hand side of this equation represents the temporal change of the salinity variance in time, referred to as the _tendency_. The first three terms on the right-hand side - _advection, straining_ and _dissipation_, respectively - can be used to quantify and visualise the mechanisms responsible for the spatial patterns of stratification within a coast-delta system and thus give us insight into the system's behaviour. This package can be used to calculate these four different terms based on the provided model data.

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
The core of the package is built around a constructor class (`constructorSVA`) that initialises the analysis environment from either a file path or an in-memory dataset. It serves as the entry point for hydrodynamic and tracer analyses. This design ensures flexibility in handling different data sources, such as `xarray.Dataset`, `xarray.DataArray`, `xugrid.UgridDataset`, `xugrid.UgridDataArray`, or other UGRID-compliant data structures.

The constructor accepts as input either a NetCDF file or an [xarray](https://docs.xarray.dev/en/stable/)-based (or [xugrid](https://deltares.github.io/xugrid/api.html)-based) object, together with a dictionary describing the relevant physical variables (e.g., velocity components, tracer, volume, flow area). Upon initialisation, the class reads the dataset. It automatically attaches references to fundamental hydrodynamic variables (velocity components, volume, viscosity, tracer concentration, depth), as well as optional fields such as layer interfaces. Grid-related attributes (node/edge/face coordinates, connectivity, and dimensional metadata) are stored for subsequent computations. This design abstracts away dataset-specific details and provides a standardised interface for subsequent diagnostics and numerical experiments.

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
