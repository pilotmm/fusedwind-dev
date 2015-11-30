from openmdao.api import Component, Group
import numpy as np

class WeibullCDF(Component):
    """Weibull cumulative distribution function"""
    def __init__(self, length):
        super(WeibullCDF, self).__init__()

        # Inputs
        self.add_param('A', shape=1, desc='scale factor')
        self.add_param('k', shape=1, desc='shape or form factor')
        self.add_param('x', shape=length, desc='input curve')
    
        # Outputs
        self.add_output('CDF', shape=length, desc='probabilities out')

    def solve_nonlinear(self, params, unknowns, resids):
        A = params['A']
        k = params['k']
        x = params['x']

        unknowns['CDF'] = 1.0 - np.exp(-(x/A)**k)

    def linearize(self, params, unknowns, resids):
        A = params['A']
        k = params['k']
        x = params['x']
           
        d_CDF_d_x = np.diag(- np.exp(-(x/A)**k) * (1./A) * (-k * ((x/A)**(k-1.0))))
        d_CDF_d_A = - np.exp(-(x/A)**k) * (1./x) * (k * ((A/x)**(-k-1.0)))
        d_CDF_d_k = - np.exp(-(x/A)**k) * -(x/A)**k * np.log(x/A)

        J = {}
        J['CDF', 'x'] = d_CDF_d_x
        J['CDF', 'A'] = d_CDF_d_A
        J['CDF', 'k'] = d_CDF_d_k

        return J
    
class AEPComponent(Component):
    """ Basic component for aep estimation for an entire wind plant with the wind resource and single turbine power curve as inputs."""
    def __init__(self, length):
        super(AEPComponent, self).__init__()
    
        # Inputs
        self.add_param('CDF', shape=length, desc='probabilities in')
        self.add_param('power_curve', shape=length, units='W', desc='power curve (power)')
        self.add_param('machine_rating', shape=1, units='kW', desc='machine power rating')
    
        # parameters
        self.add_param('array_losses', val=0.059, desc='energy losses due to turbine interactions - across entire plant')
        self.add_param('other_losses', val=0.0, desc='energy losses due to blade soiling, electrical, etc') 
        self.add_param('availability', val=0.94, desc='average annual availbility of wind turbines at plant')
        self.add_param('turbine_number', val=100, desc='total number of wind turbines at the plant')
    
        # outputs
        self.add_output('gross_aep', shape=1, desc='Gross Annual Energy Production before availability and loss impacts', units='kW*h')
        self.add_output('net_aep', shape=1, desc='Net Annual Energy Production after availability and loss impacts', units='kW*h')
        self.add_output('capacity_factor', shape=1, desc='plant capacity factor')

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['gross_aep'] = params['turbine_number'] * np.trapz(params['power_curve'], params['CDF'])*365.0*24.0  # in kWh
        unknowns['net_aep'] = params['availability'] * (1-params['array_losses']) * (1-params['other_losses']) * unknowns['gross_aep']
        unknowns['capacity_factor'] = unknowns['net_aep'] / (365.0*24.0 * params['machine_rating'] * params['turbine_number'])

    def linearize(self, params, unknowns, resids):

        P = params['power_curve']
        CDF = params['CDF']
        factor = params['availability'] * (1-params['array_losses']) * (1-params['other_losses'])*365.0*24.0 * params['turbine_number']

        n = len(P)
        dAEP_dP = np.gradient(CDF)
        dAEP_dP[0] /= 2
        dAEP_dP[-1] /= 2
        d_gross_d_p = dAEP_dP * 365.0 * 24.0 * params['turbine_number']
        d_net_d_p = dAEP_dP * factor

        dAEP_dCDF = -np.gradient(P)
        dAEP_dCDF[0] = -0.5*(P[0] + P[1])
        dAEP_dCDF[-1] = 0.5*(P[-1] + P[-2])
        d_gross_d_cdf = dAEP_dCDF * 365.0 * 24.0 * params['turbine_number']
        d_net_d_cdf = dAEP_dCDF * factor

        #loss_factor = self.availability * (1-self.array_losses) * (1-self.other_losses)

        #dAEP_dlossFactor = np.array([self.net_aep/loss_factor])
        J = {}
        J['gross_aep', 'CDF_V'] = d_gross_d_cdf
        J['gross_aep', 'power_curve'] = d_gross_d_p
        J['net_aep', 'CDF_V'] = d_net_d_cdf
        J['net_aep', 'power_curve'] = d_net_d_p
        #self.J[0, 2*n] = dAEP_dlossFactor
        return J


class AEPWeibullGroup(Group):
    """ Basic group for aep estimation for an entire wind plant with the wind resource and single turbine power curve as inputs."""
    def __init__(self, length):
        super(AEPWeibullGroup, self).__init__()

        self.add('cdf', WeibullCDF(length), promotes=['*'])
        self.add('aep', AEPComponent(length), promotes=['*'])






























