Theory: Salinity Variance Budget
=================================

This chapter describes the theoretical framework underlying pySVA, based on the work of Burchard et al. (2008) and Li et al. (2018).

Total Salinity Variance Equation
--------------------------------

Burchard et al. (2008) derived the total salinity variance equation starting from the Reynolds-averaged salt conservation advection-diffusion equation in a three-dimensional domain:

.. math::

    \frac{\partial S}{\partial t} + \mathbf{u} \cdot \nabla S - \nabla \cdot (\mathbf{K} \nabla S) = 0

where:

- :math:`\mathbf{u} = (u, v, w)` is the three-dimensional velocity vector
- :math:`\mathbf{K} = (K_{xx} + K_{yy} + K_{zz})` is the diffusivity tensor

Applying Reynolds decomposition, the total salinity and velocity can be decomposed into a volume mean and a varying part:

.. math::

    S = [[S]] + S'_{tot}, \quad \mathbf{u} = [[\mathbf{u}]] + \mathbf{u}'

where :math:`[[\cdot]]` denotes the volume average.

Substituting this decomposition and manipulating the equations yields the salinity variance equation:

.. math::

    \frac{\partial(S'_{tot})^2}{\partial t} + \nabla\cdot[\mathbf{u}(S'_{tot})^2 - \mathbf{K}\nabla(S'_{tot})^2] - 2S'[[\mathbf{u}' \cdot \nabla S'_{tot}]] = -2(\mathbf{K}\nabla S'_{tot}) \cdot (\nabla S'_{tot})

Vertical Salinity Variance
---------------------------

Li et al. (2018) extended this framework by decomposing the total salinity variance into horizontal and vertical contributions. For a single vertical water column, decomposing salinity and velocity as:

.. math::

    S = \overline{S} + S'_v, \quad \mathbf{u} = \mathbf{\overline{u}} + \mathbf{u}'_v

where the overline denotes depth averaging, the vertical salinity variance equation becomes:

.. math::

    \begin{split}
    \underbrace{\frac{\partial}{\partial t}\int(S'_v)^2 \mathrm{d}z}_{\text{tendency}} = -&\underbrace{\nabla_h \cdot \int \mathbf{u_h}(S'_v)^2 \mathrm{d}z}_{\text{advection}} + \underbrace{\int -2\mathbf{u'_v}S'_v \cdot \nabla\overline{S} \mathrm{d}z}_{\text{straining}} \\
    &- \underbrace{\int 2\kappa_{zz}\bigg(\frac{\partial S}{\partial z}\bigg)^2 \mathrm{d}z}_{\text{dissipation}} - \underbrace{\int\mathcal{M}_{\mathrm{num}} \mathrm{d}z}_{\text{numerical mixing}}
    \end{split}

Variance Decomposition
----------------------

The total salinity variance can be decomposed into horizontal and vertical components:

.. math::

    (S'_v)^2 &= (S - \overline{S})^2 \quad \text{(vertical variance)} \\
    (S'_h)^2 &= (\overline{S} - [[S]])^2 \quad \text{(horizontal variance)} \\
    (S'_{tot})^2 &= (S - [[S]])^2 \quad \text{(total variance)}

Budget Terms
------------

The variance budget equation contains four main terms:

1. **Tendency**: :math:`\frac{\partial}{\partial t}\int(S'_v)^2 \mathrm{d}z`
   - Temporal change of variance
   - Left-hand side of the equation

2. **Advection**: :math:`\nabla_h \cdot \int \mathbf{u_h}(S'_v)^2 \mathrm{d}z`
   - Transport of variance by mean flow
   - Quantifies horizontal redistribution of mixed water

3. **Straining**: :math:`\int -2\mathbf{u'_v}S'_v \cdot \nabla\overline{S} \mathrm{d}z`
   - Variance production/destruction by vertical mixing and shear
   - Related to internal wave activity

4. **Dissipation**: :math:`\int 2\kappa_{zz}(\partial S/\partial z)^2 \mathrm{d}z`
   - Variance reduction by turbulent diffusion
   - Related to molecular and turbulent mixing

Understanding these terms enables researchers to identify which processes dominate stratification patterns in coastal-delta systems.

References
----------

[1] Burchard, H., & Rennau, H. (2008). Comparative quantification of physically and numerically induced mixing in ocean models. *Ocean Modelling*, 20(3), 293-311. https://doi.org/10.1016/j.ocemod.2007.10.003

[2] Li, X., Geyer, W. R., Zhu, J., & Wu, H. (2018). The Transformation of Salinity Variance: A New Approach to Quantifying the Influence of Straining and Mixing on Estuarine Stratification. *Journal of Physical Oceanography*, 48(3), 607-623. https://doi.org/10.1175/JPO-D-17-0189.1
