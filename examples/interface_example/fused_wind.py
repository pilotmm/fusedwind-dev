
try:
    from openmdao.api import Component
except:
    from dummy_open_mdao import Component

import numpy as np
import copy

# Lets define some fused wind variables
#######################################

# This is the number of wind speeds used in a calculation
fvar_nwsp =  { 'name': 'fvar_nwsp' , 'type': int, 'val': 1 }
# A vector for each of the wind speeds used in a calculation
fvar_wind_curve = { 'name': 'fvar_wind_curve', 'units': 'm/s', 'type': np.ndarray, 'shape': [fvar_nwsp], 'val': np.array([0.])}
# A vector for eacho of the powers
fvar_power_curve = { 'name': 'fvar_power_curve', 'units': 'kW', 'type': np.ndarray, 'shape': [fvar_nwsp], 'val': np.array([0.])}


# The following are helper functions to create a custom interface
#################################################################

def create_interface():

    return {'output': {}, 'input': {}}

def set_variable(inner_dict, variable):

    inner_dict[variable['name']]=copy.deepcopy(variable)

def set_input(fifc, variable):

    set_variable(fifc['input'], variable)

def set_output(fifc, variable):

    set_variable(fifc['output'], variable)

def extend_interface(base, extension):

    for k, v in extension['input'].items():
        set_input(base, v)

    for k, v in extension['output'].items():
        set_output(base, v)

    return base

# Lets define a fused wind output interfaces
############################################

# This is the output interface for the power curve component
fifc_power_curve_output = create_interface()
set_output(fifc_power_curve_output, fvar_wind_curve)
set_output(fifc_power_curve_output, fvar_power_curve)

# The following are helper functions to help objects implement interfaces
#########################################################################

class Fused_Component(Component):

    def __init__(self):

        super(Fused_Component,self).__init__()
        self.interface = create_interface()

    def implement_fifc(self, fifc, **kwargs):

        for k, v in fifc['input'].items():

            # Apply the sizes of arrays
            if 'shape' in v.keys():
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
                if 'val' in v.keys():
                    v['val']=np.zeros(v['shape'])

            # Add our parameter
            self.add_param(**v)

        for k, v in fifc['output'].items():

            # Apply the sizes of arrays
            if 'shape' in v.keys():
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
                if 'val' in v.keys():
                    v['val']=np.zeros(v['shape'])

            # add out output
            self.add_output(**v)

    def add_param(self, **kwargs):

        set_input(self.interface, kwargs)
        Component.add_param(self, **kwargs)

    def add_output(self, **kwargs):

        set_output(self.interface, kwargs)
        Component.add_output(self, **kwargs)

class Fused_Alternate(object):

    def __init__(self):

        super(Fused_Alternate,self).__init__()
        self.interface = create_interface()

    def add_param(self, **kwargs):

        set_input(self.interface, kwargs)
        print('using add params alternate with')
        print(kwargs)

    def add_output(self, **kwargs):

        set_output(self.interface, kwargs)
        print('using add unknowns alternate with')
        print(kwargs)

    def implement_fifc(self, fifc, **kwargs):

        for k, v in fifc['input'].items():

            # Apply the sizes of arrays
            if 'shape' in v.keys():
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
                if 'val' in v.keys():
                    v['val']=np.zeros(v['shape'])

            self.add_param(**v)

        for k, v in fifc['output'].items():

            # Apply the sizes of arrays
            if 'shape' in v.keys():
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
                if 'val' in v.keys():
                    v['val']=np.zeros(v['shape'])

            self.add_output(**v)

