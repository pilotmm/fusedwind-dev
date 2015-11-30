import numpy as np
import unittest

from openmdao.api import Problem, IndepVarComp

from fusedwind.plant_flow.basic_aep import AEPWeibullGroup


def configure():
    wind_curve = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, \
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
    power_curve = np.array([0.0, 0.0, 0.0, 187.0, 350.0, 658.30, 1087.4, 1658.3, 2391.5, 3307.0, 4415.70, \
                            5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, \
                            5000.0, 5000.0, 0.0])
    
    length = len(wind_curve)
    A = 8.35
    k = 2.15
    array_losses = 0.059
    other_losses = 0.0
    availability = 0.94
    turbine_number = 100
    machine_rating = 5000.0
    
    p = Problem()
    p.root = AEPWeibullGroup(length)
    

    p.root.add('add_x', IndepVarComp('x', wind_curve),
               promotes=['x'])
    p.root.add('add_A', IndepVarComp('A', A),promotes=['A'])
    p.root.add('add_k', IndepVarComp('k', k),promotes=['k'])
    p.root.add('add_power_curve', IndepVarComp('power_curve', power_curve),
               promotes=['power_curve'])
    p.root.add('add_array_losses', IndepVarComp('array_losses', array_losses),
               promotes=['array_losses'])
    p.root.add('add_other_losses', IndepVarComp('other_losses', other_losses),
               promotes=['other_losses'])
    p.root.add('add_availability', IndepVarComp('availability', availability),
               promotes=['availability'])
    p.root.add('add_turbine_number', IndepVarComp('turbine_number', turbine_number),
               promotes=['turbine_number'])
    p.root.add('add_machine_rating', IndepVarComp('machine_rating', machine_rating),
               promotes=['machine_rating'])
    
    p.setup()

    return p

class AEPTestCase(unittest.TestCase):

    def setUp(self):
        self.p = configure()
        
    def tearDown(self):
        pass
        
    def test_aep_weibull(self):
        self.p.run()
        self.assertAlmostEqual(self.p.root.unknowns['gross_aep'], 1570713782.2, places=1)
        self.assertAlmostEqual(self.p.root.unknowns['net_aep'], 1389359168.9, places=1)
        print self.p.root.unknowns['gross_aep']
        print self.p.root.unknowns['net_aep']

        
if __name__ == "__main__":
    unittest.main()
    
    