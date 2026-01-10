Installation & Getting Started
==============================

Installation
------------

pySVA is currently under active development. The recommended way to install the most recent version is directly from GitHub:

.. code-block:: bash

    git clone https://github.com/mgeraeds/pysva.git
    cd pysva
    pip install .

This will install the package in development mode, allowing you to use the latest features and contribute improvements.

Requirements
~~~~~~~~~~~~

The package depends on:

- Python 3.7+
- xarray
- xugrid
- dfm-tools
- numpy
- dask
- xarray-einstats

Usage Overview
--------------

Core Architecture
~~~~~~~~~~~~~~~~~

The core of pySVA is built around the **constructorSVA** class, which:

1. **Initializes** the analysis environment from either a file path or in-memory dataset
2. **Extracts** grid topology, coordinates, and dimensional metadata
3. **Provides** a standardized interface for variance budget computations

This design ensures flexibility in handling different data sources while abstracting away dataset-specific details.

Input Formats
~~~~~~~~~~~~~

The ``constructorSVA`` class accepts:

- **File paths** to NetCDF files
- **xarray.Dataset** objects
- **xarray.DataArray** objects
- **xugrid.UgridDataset** objects (UGRID-compliant data structures)

Data Description Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When initializing the class, you must provide a ``data_description`` dictionary that maps variable names to dataset keys.

**Required keys:**

- ``'velx'``: X-component of velocity
- ``'vely'``: Y-component of velocity
- ``'velz'``: Z-component of velocity
- ``'viscosity'``: Eddy viscosity
- ``'tracer'``: Tracer concentration (e.g., salinity)
- ``'depth'``: Water depth

**Optional keys:**

- ``'horizontal_diffusivity'``: Horizontal diffusion coefficient
- ``'interfaces'``: Layer interface elevations
- ``'bed_level'``: Bed elevation
- ``'volume'``: Cell volume
- ``'flow_area'``: Cross-sectional flow area

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    from pySVA.sva_core import constructorSVA
    from pySVA.sva_calc import compute_tracer_variance, compute_advection
    
    # Initialize the SVA object
    data_desc = {
        'velx': 'mesh2d_u1_x',
        'vely': 'mesh2d_u1_y',
        'velz': 'mesh2d_w',
        'viscosity': 'mesh2d_vicwwu',
        'tracer': 'mesh2d_sa1',
        'depth': 'mesh2d_flowelem_bl',
        'interfaces': 'mesh2d_interface_zm'
    }
    
    sva = constructorSVA('path/to/file.nc', data_desc)
    
    # Compute variance terms
    tracer_variance = compute_tracer_variance(sva)
    advection_term = compute_advection(sva)

Architecture Diagram
~~~~~~~~~~~~~~~~~~~~

::

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
    ┌──────────────────────────────┐
    │   Cached Geometry Properties │
    │ ──────────────────────────── │
    │ • flow_area                  │
    │ • face_area                  │
    │ • water_depth                │
    │ • bed_level                  │
    │ • edge_length                │
    │ • face_edge_weights          │
    │ • edge_face_weights          │
    │ • Coordinates & connectivity │
    └───────────────┬──────────────┘
                    │
                    ▼
    ┌──────────────────────────────┐
    │    Variance Budget Terms     │
    │ ──────────────────────────── │
    │ • Tracer variance            │
    │ • Advection                  │
    │ • Straining                  │
    │ • Dissipation                │
    └──────────────────────────────┘

More Examples
-------------

For detailed examples and tutorials, see the ``notebooks/`` directory in the repository.

License
-------

This software is licensed under the MIT License. See the LICENSE file for details.

How to Cite
-----------

When using pySVA in your research, please cite:

    Geraeds, M. (2025). pySVA: Python-based toolbox for salinity variance budget computations on unstructured model output. 4TU.ResearchData. Software. https://doi.org/10.4121/66de21de-b6b7-462d-8a50-dbcf4f858b64

BibTeX format:

.. code-block:: bibtex

    @misc{geraeds_pysva,
        title     = {pySVA: Python-based toolbox for salinity variance budget computations on unstructured model output},
        author    = {Geraeds, M.},
        publisher = {4TU.ResearchData},
        year      = {2025},
        version   = {1.0},
        doi       = {10.4121/66de21de-b6b7-462d-8a50-dbcf4f858b64},
    }
