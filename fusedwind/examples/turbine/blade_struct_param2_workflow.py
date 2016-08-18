
import unittest
import numpy as np
import os
import pkg_resources

from openmdao.api import Group, Problem

from PGL.main.distfunc import distfunc
from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure, \
                                        SplinedBladeStructureParam2, \
                                        BladeStructureProperties
from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       SplinedBladePlanform, \
                                       PGLRedistributedPlanform

PATH = pkg_resources.resource_filename('fusedwind', 'turbine/test')

# --- 1 ---

nsec = 20
st3dn = read_bladestructure(os.path.join(PATH, 'data_version_2/Param2_10MW'))
p = Problem(root=Group())
root = p.root
pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))

nsec_ae = 30
nsec_st = 20
dist = np.array([[0., 1./nsec_ae, 1], [1., 1./nsec_ae/3., nsec_ae]])
s_ae = distfunc(dist)
s_st = np.linspace(0, 1, nsec_st)
pf = redistribute_planform(pf, s=s_ae)

# add planform spline component
spl_ae = root.add('pf_splines', SplinedBladePlanform(pf), promotes=['*'])

# --- 2 ---

# component for interpolating planform onto structural mesh
redist = root.add('pf_st', PGLRedistributedPlanform('_st', nsec_ae, s_st), promotes=['*'])

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

# --- 3 ---

spl = p.root.add('st_splines', SplinedBladeStructureParam2(st3dn, (200, nsec, 3)), promotes=['*'])

p.root.add('st_props', BladeStructureProperties((200, nsec, 3), st3dn,
                                                [3, 6, 9, 12]),
                                                promotes=['*'])

spl.add_spline(('cap_width_ss',
                'cap_width_ps'),
                np.array([0., 0.25, 0.5, 0.75, 1.]), spline_type='bezier')
p.setup()

# --- 4 ---

p.run()
import matplotlib.pylab as plt
plt.plot(p['s_st'], p['r04_width'] +
                    p['r05_width'] +
                    p['r06_width'], 'r-', label='Original')

# modify three of the cap width CPs
p['cap_width_ss_C'][:3] += 0.5 / 86.366
p.run()
plt.plot(p['s_st'], p['r04_width'] +
                    p['r05_width'] +
                    p['r06_width'], 'b-', label='Modified')
plt.legend(loc='best')
plt.xlabel('r/R [-]')
plt.ylabel('Cap width [-]')
plt.show()
