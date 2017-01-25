
import copy
from wind_io import fifc_power_curve_output \
		, create_interface \
		, extend_interface \
		, Fused_Component \
		, Fused_Alternate

my_power_curve_output = create_interface()
my_power_curve_output = extend_interface(my_power_curve_output , fifc_power_curve_output)

class wrapped_power_curve(Fused_Component):

	def __init__(self):

		super(type(self),self).__init__()
		self.implement_fifc(my_power_curve_output, fvar_nwsp=15)
		self.add_param(name='what')

	def solve_nonlinear(self, params, unknowns, resids):

		print("running solve nonlear")

if __name__ == '__main__':

	my_obj = wrapped_power_curve()
	my_obj.solve_nonlinear({} , {} , {})
	print(my_obj.interface)

	def Use_Alternate(my_class):

		return type(my_class.__name__, (Fused_Alternate,), dict(my_class.__dict__))

	my_Class = Use_Alternate(wrapped_power_curve)
	print(type(my_Class))
	print(my_Class.__dict__)
	print(my_Class.__bases__)
	my_obj = my_Class()
	my_obj.solve_nonlinear({} , {} , {})
	print(my_obj.interface)

