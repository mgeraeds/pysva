Overview
========

**pySVA** is a Python-based open-source toolbox for computing terms in a salinity variance budget on unstructured mesh model output. It applies to model output from models with unstructured, staggered C-grid meshes, particularly from models like `D-FLOW Flexible Mesh <https://content.oss.deltares.nl/delft3d/D-Flow_FM_User_Manual.pdf>`_.

The toolbox implements the salinity variance budget framework of `Burchard et al. (2008) <https://doi.org/10.1016/j.ocemod.2007.10.003>`_ and `Li et al. (2018) <https://doi.org/10.1175/JPO-D-17-0189.1>`_, which decomposes tracer variance changes into contributions from advection, straining/mixing, and dissipation. This approach enables quantification of both physical and numerical mixing processes in hydrodynamic models.

It is available as open source on `Github <https://github.com/mgeraeds/pysva>`_ under the MIT license.

Purpose
-------

The pySVA toolbox enables quantification of physical and numerical mixing processes in hydrodynamic models by decomposing changes in tracer variance into contributions from:

- **Advection**: Transport of variance by mean flow
- **Straining**: Variance production/destruction due to vertical mixing and shear
- **Dissipation**: Variance reduction due to diffusion (turbulent mixing)

This approach provides insight into the mechanisms responsible for spatial patterns of stratification in coastal and estuarine systems.

Key Features
------------

- Handles multiple input formats: NetCDF files, xarray Datasets, and xugrid UgridDatasets
- Automatically extracts grid topology and coordinate systems
- Provides cached geometric properties for efficient computation
- Implements the variance budget framework of Burchard et al. (2008) and Li et al. (2018)

Status
------

⚠️ **Note:** This package is currently in limited maintenance mode. Issues can still be opened, but fixes might take longer.
