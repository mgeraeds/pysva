.. pySVA documentation master file, created by
   sphinx-quickstart on Sat Jan  3 13:32:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pySVA documentation
===================

**pySVA** is a Python-based open-source toolbox for computing terms in a salinity variance budget on unstructured mesh model output, particularly from models like `D-FLOW Flexible Mesh <https://content.oss.deltares.nl/delft3d/D-Flow_FM_User_Manual.pdf>`_.

The toolbox implements the salinity variance budget framework of `Burchard et al. (2008) <https://doi.org/10.1016/j.ocemod.2007.10.003>`_ and `Li et al. (2018) <https://doi.org/10.1175/JPO-D-17-0189.1>`_, which decomposes tracer variance changes into contributions from advection, straining/mixing, and dissipation. This approach enables quantification of both physical and numerical mixing processes in hydrodynamic models.

**Citation:**

Geraeds, M. (2025). pySVA: Python-based toolbox for salinity variance budget computations on unstructured model output. 4TU.ResearchData. Software. https://doi.org/10.4121/66de21de-b6b7-462d-8a50-dbcf4f858b64

BibTeX:

.. code-block:: bibtex

    @misc{geraeds_pysva,
        title     = {pySVA: Python-based toolbox for salinity variance budget computations on unstructured model output},
        author    = {Geraeds, M.},
        publisher = {4TU.ResearchData},
        year      = {2025},
        version   = {1.0},
        doi       = {10.4121/66de21de-b6b7-462d-8a50-dbcf4f858b64},
    }

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

   overview
   theory
   getting_started
   api

