
import numpy as np

def calculate_power_curve(n_wsp, min_wsp, max_wsp, air_density, turbulence_int, rated_power, rotor_diameter, max_Cp, gearloss_const, gearloss_var, genloss, convloss, max_tipspeed):

    rated_power *= 1.e6

    wsp = np.linspace(min_wsp, max_wsp, n_wsp)
    P_aero = 0.5 * air_density * (np.pi * (rotor_diameter/2.)**2) * wsp**3 * max_Cp

    P_gear = np.maximum(0., P_aero - gearloss_const * rated_power - P_aero * gearloss_var)
    P_gen = P_gear * (1 - genloss)
    P_conv = P_gen * (1 - convloss)
    P_raw = np.minimum(P_conv, rated_power)

    # TurbulenceCorrection
    NormDist = np.zeros((n_wsp-1, n_wsp))

    sigma = wsp*turbulence_int
    NDist = lambda x, my, std: (1./(np.sqrt(2*np.pi)*std))*np.exp(-(x-my)*(x-my)/(2.*std**2))

    ProbSum = np.zeros(P_conv.shape[0])
    for i in range(n_wsp - 1):
     for j in range(n_wsp):
         NormDist[i, j] = NDist(wsp[j], wsp[i+1], sigma[i+1])
         ProbSum[i] = ProbSum[i] + NormDist[i, j]

    P_turb = np.zeros(n_wsp)
    for i in range(1, n_wsp):
        for j in range(n_wsp):
            P_turb[i] = P_turb[i] + P_raw[j] * NormDist[i-1, j] / ProbSum[i-1]

    ideal_power_curve = P_raw * 1.e-3
    power_curve = P_turb * 1.e-3
    wind_curve = wsp

    # estimate rated torque 07/09/2015
    wsp_index = list(P_raw).index(rated_power)
    rated_wind_speed = wind_curve[wsp_index]
    rated_speed = (max_tipspeed/(0.5*rotor_diameter)) * (60.0 / (2*np.pi))
    rated_torque = rated_power/(((1-genloss)*(1-convloss)*(1-gearloss_const-gearloss_var))*(rated_speed*(np.pi/30.)))

    return rated_wind_speed, ideal_power_curve, power_curve, wind_curve, rated_torque, rated_speed

if __name__ == '__main__':

    n_wsp = 16
    min_wsp = 5.0
    max_wsp = 20.0
    air_density = 1.2
    turbulence_int = 0.1
    rated_power = 2.0
    rotor_diameter = 90.0
    max_Cp = 0.45
    gearloss_const = 0.01
    gearloss_var = 0.01
    genloss = 0.01
    convloss = 0.01
    max_tipspeed = 60.0

    rated_wind_speed, ideal_power_curve, power_curve, wind_curve, rated_torque, rated_speed = calculate_power_curve(n_wsp, min_wsp, max_wsp, air_density, turbulence_int, rated_power, rotor_diameter, max_Cp, gearloss_const, gearloss_var, genloss, convloss, max_tipspeed)

    print 'rated_wind_speed:'
    print rated_wind_speed
    print 'ideal_power_curve:'
    print ideal_power_curve
    print 'power_curve:'
    print power_curve
    print 'wind_curve:'
    print wind_curve
    print 'rated_torque:'
    print rated_torque
    print 'rated_speed:'
    print rated_speed

