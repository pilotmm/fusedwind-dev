
.. _sec_blade_geometry_ex_label:

Blade Geometry
++++++++++++++

A blade planform in FUSED-Wind is described by spanwise distributions of the variables *chord*, *twist*, *relative thickness*, and *pitch axis aft leading edge*.
The default file format used to represent a blade planform contains the following columns:

.. literalinclude:: ../fusedwind/turbine/test/data/DTU_10MW_RWT_blade_axis_prebend.dat
   :lines: 1

*x* is defied positive towards the trailing edge, *y* positive towards the blade suction side, and *z* running along the blade axis from root to tip.
Following the main axis, there are three rotations, where the blade twist is specified in *rot_z*.
The inclusion of *rot_x* and *rot_y* allows for an arbitrary orientation of the cross section.
By default *rot_x* and *rot_y* are set to zero, and it is assumed that the cross sections are locally normal to the blade axis.
Next, the *chord* and relative thickness *rthick* are specified, note that *rthick* is defined as a number in the range [0, 1].
However, the range [0, 100] is also accepted and will be normalised to 1 internally in the reader.
The pitch axis aft leading edge is the final parameter, which specifies the normalised chordwise (*x*) distance between the leading edge and the main axis.

A lofted shape can be generated from this planform in combination with a series of airfoils that are interpolated according a given interpolator, typically the relative thicknesses of the airfoils.
FUSED-Wind provides an interface to the external geometry tool called `PGL`, which optionally can be installed as a dependency when installing FUSED-Wind.
The class ``fusedwind.turbine.geometry.PGLLoftedBladeSurface`` provides an OpenMDAO interface to the ``LoftedBladeSurface`` class in `PGL`.

In the example below we show how to setup a splined blade planform and lofted blade surface. In the example, the blade planform data and the lofted surface will be discretised differently, one meant for the aerodynamic calculation, and the other for the structural.
The example is located in ``fusedwind/examples/turbine/loftedsurface_with_cps.py``.

The first step is to read in the blade planform using the :class:`fusedwind.turbine.geometry.read_bladeplanform` method.
Using the :class:`fusedwind.turbine.geometry.redistribute_planform` method, the planform is redistributed according to the desired number of points.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 1
    :end-before: # --- 2

Next, we add the :class:`fusedwind.turbine.geometry.SplinedBladePlanform` class is added,
which inherits from the :class:`openmdao.core.group.Group` class,
and has methods to define which planform parameters to add splines to.
:class:`fusedwind.turbine.geometry.PGLRedistributedPlanform` class is a simple class
that redistributes the planform according to another distribution, in our case
the structural discretisation.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 2
    :end-before: # --- 3

The next step is to define the base airfoils used to generate the lofted blade shape.
The class :class:`fusedwind.turbine.geometry.PGLLoftedBladeSurface`` provides an OpenMDAO interface to the ``LoftedBladeSurface`` class in `PGL.
This class takes several options, which we pass as a dictionary when instantiating the class.
The base airfoils are automatically redistributed according to the same distribution
function by PGL, so the base airfoils can be distributed differently.
The `blend_var` argument tells PGL how to weight the airfoils when blending them.
By default this will be done according to relative thickness, but you can also
specify airfoils to be at specific spanwise locations resulting in interpolation
weighted according to span. See the class documentation for all the options.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 3
    :end-before: # --- 4

In the final step in this example we specify which parameters to add a spline to
using the `add_spline` method.
Adding a spline adds an `IndepVarComp` with an array of spline CPs, which will be named
`<varname>_C`, i.e. `chord_C` for the chord.
The `SplinedBladePlanform` class currently only supports FFD splines, but you can
choose to use either a Bezier or pchip basis spline.
The former results in smoother and more global variations where the latter
allows for local control.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 4
    :end-before: # --- 5

Finally, we can run the example.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 5
    :end-before: # --- 6

To add sweep to the blade, we need to perturb the `x_C` control points array.
The plot shows the effect on the surface.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 6
    :end-before: # --- 7

Modifying the chord is done in the same way by changing the `chord_C` array as shown below.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 7
    :end-before: # --- 8

.. _bladesurface_planform-fig:

    .. image::  images/chord.png
       :width: 49 %
    .. image::  images/twist.png
       :width: 49 %
    .. image::  images/rthick.png
       :width: 49 %
    .. image::  images/p_le.png
       :width: 49 %


.. _bladesurface_lofted-blade-fig:

.. figure:: /images/lofted_blade.png
    :width: 70 %
    :align: center

    Lofted blade shape.


.. _bladesurface_topview-fig:

.. figure:: /images/bladesurface_topview.png
    :width: 70 %
    :align: center

    Top view of lofted blade surface with sweep.


.. _bladeplanform_spline-fig:

.. figure:: /images/chord_ffd_spline.*
    :width: 80 %
    :align: center

    Blade chord pertubation.


Blade Structure Example
+++++++++++++++++++++++

The blade structure parameterization is primarily aimed for conceptual analysis and optimization,
where the geometric detail is fairly low to enable its use more efficiently in an optimization context.

A cross-section is divided into a number of *regions* that each cover a fraction of the cross-section.
Each region, in turn contains a stack of materials.
In each layer, the material type, thickness and layup angle can be specified.
The materials used in the blade are specified based on apparent properties of the constituent materials, which need to be pre-computed using simple micromechanics equations and classical lamination theory.

The figure below shows a blade cross section where the region division points (DPs) are indicated.
The location of each DP is specified as a normalized arc length along the cross section
starting at the trailing edge pressure side with a value of s=-1., and along the surface to the leading edge where s=0.,
along the suction side to the trailing edge where s=1.
Any number of regions can thus be specified, distributed arbitrarily along the surface.

.. _bladestructure_cross_sec-fig:

.. figure:: /images/cross_sec_sdef13.png
    :width: 80 %
    :align: center

    Blade cross section with region division points (DPs) indicated with red dots and shear webs drawn as green lines.

The spar caps are specified in the same way as other regions, which means that their widths and position along the chord are not default parameters.
It is only possible to place shear webs at the location of a DP, which means that a single shear web topology would require the spar cap to be split into two regions.

The full blade parameterization is a simple extrusion of the cross-sectional denition, where every region covers the entire span of the blade.
The DP curves marked with red dots in the plot below are simple 1-D arrays as function of span that as in the cross-sectional definition take a value between -1. and 1.
The distribution of material and their layup angles along the blade are also specified as simple 1-D arrays as function of span.
Often, a specific composite will not cover the entire span, and in this case the thickness of this material is simply specified to be zero at that given spanwise location.

.. _bladestructure_lofted-blade-fig:

.. figure:: /images/structural_cross_sections.png
    :width: 15cm
    :align: center

    Lofted blade with region division points indicated with red dots and shear webs drawn as green lines.

FUSED-Wind provides a simple file format for storing the structural definition of the blade.
The structure is defined in a set of files with the same *base-name*, each described below. You can find an example using this file format in fusedwind/turbine/test/data/DTU10MW*.

The *<base-name>.mat* file contains the properties of all the materials used in the blade.
The first line in the header contains the names of each of the materials.
The second line lists the names of each of the parameters defining the materials,
followed by the data with the same number of lines as materials listed in line 1 (not shown below):

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW.mat
   :lines: 1-2

The *<base-name>.failmat* file contains the strength properties of all the materials used in the blade.
The first line in the header contains the names of each of the materials.
The second line lists the names of each of the parameters defining the strength properties of the materials,
followed by the data with the same number of lines as materials listed in line 1 (not shown below):

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW.failmat
   :lines: 1-2

The failure criterium defined in column 1 is an integer value corresponding to
1:'maximum_strain', 2:'maximum_stress', 3:'tsai_wu'.

The *<base-name>.dp3d* file contains the DPs' positions along the span.
The header of the file contains the names of the webs in line 1, the connectivity of the shear webs with the surface DPs in the following lines, and finally the names of each of the regions.
The data (not shown below) contains in column 1, the running length along the blade axis, followed by the chordwise curve fraction of the DP along the span for each of the DPs.

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW.dp3d
   :lines: 1-6

The *<base-name>_<rname>.st3d* files contain the thickness distributions of each of the materials in the individual regions. The header contains the following: the name of the region in line 1 and the names of each of the materials in line two.
The data (not shown below) contains in column 1, the running length along the blade axis, followed by the thicknesses of each material along the span.

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW_region04.st3d
   :lines: 1-2



.. _bladestructure_spline-fig:

.. figure:: /images/turbine_structure_uniax_perturb.png
    :width: 80 %
    :align: center

    Blade spar cap uniax thickness pertubation.


Geometric Blade Structure Parameterisation
++++++++++++++++++++++++++++++++++++++++++

In addition to the parameterisation described above that uses a flexible and
normalized approach, a more intuitive parameterisation based on the lofted geometry
is available in FUSED-Wind.
This allows the user to place the spar caps, webs, leading and trailing edge
reinforments relative to a reference plane, making it easy to generate a new
structural geometry from scratch.

In addition to the structural input files described above, an input file
with extension `.geo3d` can be provided.
An example of the header of this file is shown below:

.. literalinclude:: ../fusedwind/turbine/test/data_version_2/Param2_10MW.geo3d
   :lines: 1-7

Starting from the bottom line, the primary parameters describing the structural
geometry are:

* *cap_center_ps*, *cap_center_ss*: Lower and upper spar cap centers relative to the reference plane. Positive towards leading edge.

* *cap_width_ps*, *cap_width_ss*: Lower and upper spar cap widths measured as surface curve length.

* *te_width*, *le_width*: Trailing and leading edge reinforcement widths (le_width spans across the leading edge) measured as surface curve length.

* *w01pos*, *w02pos*, *w03pos*: web positions relative to the reference plane. Positive towards leading edge.

The above quantities are all a function of unit span normalized with the blade length.
The additional parameters that need to be defined are:

* *le_DPs*: DPs enclosing the leading edge reinforcement.
* *te_DPs*: DPs enclosing the trailing edge reinforcements.
* *cap_DPs*: indices that enclose the two spar caps.
* *dominant_regions*: indices of regions that overwrite colliding regions
* *struct_angle*: The angle between the reference plane and the rotor
plane, defined positive nose up. Note that this angle is independent of the aerodynamic twist, and is purely used to place the webs and main laminates.

The schematic below shows a blade cross section with the above quantities.
The reference plane that the caps and webs are placed relative to is a vertical plane
starting at the blade root, ending at the blade tip.

.. _bladestructure_spline-fig:

.. figure:: /images/param2_schematic.png
    :width: 100 %
    :align: center

    Schematic showing the geometric parameterisation of the blade structure.

You can generate the structural geometry either as a pre-processing step to an optimization where you optimize using the `DP` parameterisation, or use this parameterisation directly in an optimization.
We firstly show how to call the ``ComputeDPsParam2`` class directly, which requires a pre-computed lofted blade surface as well as either an `st3d` dictionary with the structural inputs or that you specify these manually.
The example is located in ``fusedwind/examples/turbine/blade_struct_param2.py``.

The initial step is to import the necessary Python modules, as well as the classes
used in this example, which are ``PGL`` for generating the lofted surface,
and methods and classes from ``fusedwind.turbine.structure``:

.. literalinclude:: ../fusedwind/examples/turbine/blade_struct_param2.py
    :start-after: # --- 1
    :end-before: # --- 2

The next step is to generate the lofted surface with PGL.
You can find more documentation on PGL in the docs of this library;
here we just use it to make a simple surface using the planform
definition of the DTU 10MW RWT, and a series of airfoils with different
relative thicknesses.

.. literalinclude:: ../fusedwind/examples/turbine/blade_struct_param2.py
    :start-after: # --- 2
    :end-before: # --- 3

Now we can generate the structural geometry.
In this example, we read the structural definition from an input file,
but it is easy to generate it manually and pass directly into the class.
If you run this example in iPython, you can inspect the st3d object keys,
where you will see that you can set all the above described structural
parameters.

.. literalinclude:: ../fusedwind/examples/turbine/blade_struct_param2.py
    :start-after: # --- 3
    :end-before: # --- 4

Finally, the ``ComputeDPsParam2`` class provides some basic plot methods
to inspect the final structure, which are shown in the below figures.

.. literalinclude:: ../fusedwind/examples/turbine/blade_struct_param2.py
    :start-after: # --- 4
    :end-before: # --- 5

.. _bladestructure_param2-tip-fig:

.. figure:: /images/struct_param2_tipview.png
   :width: 100 %
   :align: center

   Blade structure generated using the ``ComputeDPsParam2`` class viewed from the tip
   in the rotor coordinate system.

Note that the third shear web is parallel to the main laminate and intersects the
trailing edge panel at approximately *r/R*=0.65, where it should stop.
Although the current parameterisation does not allow for discontinuous DP definitions
in the spanwise direction, this can be handled by collapsing the web DP onto the trailing edge panel DP.
However, this requires that the meshing code used has a check for zero thickness
regions and removes these before meshing.
If this capability is not available, set the parameter ``min_width`` to something
greater than zero.
It is recommended that the ``te_DPs` and ``cap_DPs`` are specified as so-called ``dominant_regions``,
which means that other regions are moved to avoid negative widths.
For regions not part of the ``dominant_regions`` coliding region edges (DPs) are moved to the mid-point between the two.

.. _bladestructure_param2-top-fig:

.. figure:: /images/struct_param2_topview.png
     :width: 100 %
     :align: center

     Blade structure generated using the ``ComputeDPsParam2`` class viewed from the suction side
     in the rotor coordinate system.

You can easily modify the inputs manually. Below we shift the main laminates and webs
forward and set the strucural angle to 15 deg.:

.. literalinclude:: ../fusedwind/examples/turbine/blade_struct_param2.py
    :start-after: # --- 5
    :end-before: # --- 6

with the following effect:


.. _bladestructure_param2-top-fig:

.. figure:: /images/struct_param2_tipview_struct_angle15.png
     :width: 100 %
     :align: center

     Plot of a blade structure with the main laminates moved forward and angled 15 degrees
     relative to the rotor plane.

In the above figure, it is also evident that some regions have collapsed to zero
width on the outer section.
The leading panels between the spar caps and the leading edge reinforcement have
been displaced by the spar caps, and the similarly the trailing edge web DPs
have collapsed onto the trailing edge DPs.
Note that the plotting functions in ``ComputeDPsParam2`` plots only valid
regions, although they still exist in the parameterisation.

To use the ``ComputeDPsParam2`` class as part of an OpenMDAO workflow,
wrapper classes are provided in ``fusedwind.turbine.structure``.

The example in ``fusedwind/examples/turbine/blade_struct_param2_workflow.py``
shows how to set this up with a splined planform and structural geometry,
and add design variables to control the structural geometry in an
optimization context.
In the example, we add splines to the pressure and suction side spar cap width
inputs, and modify the control points ``cap_width_ss_C`` which controls both splines.

.. literalinclude:: ../fusedwind/examples/turbine/blade_struct_param2_workflow.py

Running the example should produce the following plot showing the increased cap width.

.. figure:: /images/struct_param2_cap_width_C.png
     :width: 100 %
     :align: center

     Plot of the spar cap width modified using an FFD Bezier spline.
