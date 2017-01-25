
import numpy as np

def calculate_aep(wind_curve, power_curve, WeibullInput, MeanWSP, WeiA_input, WeiC_input, NYears):

    wsp = wind_curve
    n_wsp = power_curve.shape[0]
    P_turb = power_curve

    if not WeibullInput:
        WeibullC = 2.0
        WeibullA = 1.1284*MeanWSP
    else:
        WeibullC = WeiC_input
        WeibullA = WeiA_input

    NHours = np.zeros(n_wsp)
    NHours[0] = 8760*((1-np.exp(-np.exp(WeibullC*np.log((wsp[0]+0.5)/WeibullA)))))
    for i in range(1, n_wsp):
        NHours[i] = 8760. * ((1 - np.exp(-np.exp(WeibullC * np.log((wsp[i] + 0.5) / WeibullA)))) - \
                             (1 - np.exp(-np.exp(WeibullC * np.log((wsp[i] - 0.5) / WeibullA)))))

    aep_pr_wsp = P_turb * NHours
    AEPSum = np.sum(aep_pr_wsp) * 1.e-3

    aep = AEPSum
    total_aep = AEPSum * NYears

    return aep, total_aep

if __name__ == '__main__':

    wind_curve = np.array([ 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    power_curve = np.array([ 0., 351.62255477, 569.19499369, 859.23025896, 1222.86610494, 1594.88880078, 1848.45395365, 1957.40534917, 1990.04260176, 1997.87694758, 1999.55921696, 1999.9068899, 1999.97921452, 1999.99485322, 1999.99851331, 1999.99948404 ])
    WeibullInput = False
    MeanWSP = 8.0
    WeiA_input = 8.5
    WeiC_input = 2.0
    NYears = 20.0

    aep, total_aep = calculate_aep(wind_curve, power_curve, WeibullInput, MeanWSP, WeiA_input, WeiC_input, NYears)

    print 'aep'
    print aep
    print 'total_aep'
    print total_aep

