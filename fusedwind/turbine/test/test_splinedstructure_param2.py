
import unittest
import numpy as np
import os
import pkg_resources

from openmdao.api import Group, Problem

from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure, \
                                        SplinedBladeStructureParam2, \
                                        BladeStructureProperties
from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       PGLRedistributedPlanform

PATH = pkg_resources.resource_filename('fusedwind', 'turbine/test')


def configure_with_surface():

    nsec = 20
    st3dn = read_bladestructure(os.path.join(PATH, 'data_version_2/Param2_10MW'))
    p = Problem(root=Group())

    pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))
    s_new = np.linspace(0, 1, nsec)
    pf = redistribute_planform(pf, s=s_new)

    cfg = {}
    cfg['redistribute_flag'] = False
    cfg['blend_var'] = np.array([0.241, 0.301, 0.36, 1.0])
    afs = []
    for f in [os.path.join(PATH, 'data/ffaw3241.dat'),
              os.path.join(PATH, 'data/ffaw3301.dat'),
              os.path.join(PATH, 'data/ffaw3360.dat'),
              os.path.join(PATH, 'data/cylinder.dat')]:

        afs.append(np.loadtxt(f))
    cfg['base_airfoils'] = afs
    d = PGLLoftedBladeSurface(cfg, size_in=nsec, size_out=(200, nsec, 3), suffix='_st')
    r = p.root.add('blade_surf', d, promotes=['*'])

    spl = p.root.add('st_splines', SplinedBladeStructureParam2(st3dn, (200, nsec, 3)), promotes=['*'])

    p.root.add('st_props', BladeStructureProperties((200, nsec, 3), st3dn,
                                                    [3, 6, 9, 12]),
                                                    promotes=['*'])

    spl.add_spline(('cap_width_upper',
                    'cap_width_lower'),
                    np.linspace(0, 1, 4), spline_type='linear')
    spl.configure()
    p.setup()
    for k, v in pf.iteritems():
        if k+'_st' in p.root.blade_surf.params.keys():
            p[k+'_st'] = v
    return p


class TestSplinedBladeStructure(unittest.TestCase):

    def test_splines(self):
        cap_width = np.linspace(1.1, 0.8, 20) / 86.366
        cap_width[-1] = 0.25 / 86.366
        le_width = np.linspace(1.6, 0.8, 20) / 86.366
        le_width[-2] = 0.6 / 86.366
        le_width[-1] = 0.1 / 86.366
        p = configure_with_surface()
        p.run()
        self.assertEqual(np.testing.assert_array_almost_equal(p['r04_width'] +
                                                              p['r05_width'] +
                                                              p['r06_width'],
                                                              cap_width,
                                                              decimal=6), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['r08_width'],
                                                              le_width,
                                                              decimal=6), None)
if __name__ == '__main__':

    unittest.main()
    # p = configure_with_surface()
    # p.run()
