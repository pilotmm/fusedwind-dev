In the Fused-Wind collaboration we are trying to devise a system from wrapping models. The idea is to have all models conform to standard set of methods and variables names so the models can be interchangable. The code within this folder is developed by Michael McWilliam (mimc@dtu.dk) as an example of how this can be done. The code was developed with the following use-cases and objectives:

The code is developed as if there are 3 use-cases:

	- The model developer that has no knowledge of Fused-Wind thus creates models witout any thoughts to conforming to standards
		- Examples of their work is in aep.py and power_curve.py
		- Basically these files are simple calculation routines
	- The fused-wind user is someone who wants to wrap the models and then plug them into fused-wind workflows
		- An example of this work is in wrap.py
		- Connecting to fused wind should be simple as possible
	- The fused-wind developer are the people who develop the Fused-Wind code
		- An example of this work is in fused_wind.py
		- A set of helper functions, defined workflows, variables, interfaces, etc.
	- To keep the examples light-weight there is a dummy implementation of OpenMDAO contained within dummy_open_mdao.py

The task of wrapping a model is a matter of definining an interface using a mix of fused-wind defined variables and custom user defined variables. These variabels and interfaces should be objects that can be easily manipulated with python thus dictionaries were chosen for the format. The idea is these different variables and interfaces could be defined in YAML in the wind-io module. Then code can import these YAML files as dictionaries that are then subsequently used in the fused-wind work-flows and wrapped code.

Another goal was for the wrapped objects to store their IO attributes. There are instances where data transfers should be done outside of OpenMDAO. These occur when OpenMDAO is too slow or takes too much memory. In these special cases extra logic is required to manually perform all the data transfers. Having the interfaces stored on the object in a standard way allows one to write helper functions for these special cases. OpenMDAO does store interface data in a private data structure, however it is bad practice to access this data because it can be changed and thus break the code.

Due to the special cases where OpenMDAO causes problems, I wanted to have mechanism where the code can run without openMDAO. This was achieved by using meta-classes that remove the OpenMDAO inheritance.

The example given here is not complete in that the interface is not loaded from YAML files or the wrap does not actually run the models. These functions were deemed simple and straightforward and deemed not neccessary to communicate the ideas.

