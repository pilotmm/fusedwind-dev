"""
pure python example of how to generate a blade
structural geometry using PGL and the ComputeDPsParam2 class
"""

# --- 1 ---

import pkg_resources
import os
import numpy as np

from PGL.components.loftedblade import LoftedBladeSurface
from PGL.main.planform import read_blade_planform, redistribute_planform

from fusedwind.turbine.structure import ComputeDPsParam2, \
                                        read_bladestructure, \
                                        write_bladestructure

PATH = pkg_resources.resource_filename('fusedwind', 'turbine/test')

# --- 2 ---

# generate a lofted blade surface using PGL

pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))

s_st = np.linspace(0, 1, 20)

pf = redistribute_planform(pf, s=s_st)

d = LoftedBladeSurface()
d.pf = pf

# redistribute in the chordwise direction
# with 200 nodes, open TE (chord_nte=0)
# and open TE of cross sections with TE
# thickness less 2 cm
d.redistribute_flag = True
d.chord_ni = 200
d.chord_nte = 0
d.minTE = 0.000232

# privide base airfoils as function of relative thickess
d.blend_var = [0.241, 0.301, 0.36, 0.48, 1.0]
for f in [os.path.join(PATH, 'data/ffaw3241.dat'),
          os.path.join(PATH, 'data/ffaw3301.dat'),
          os.path.join(PATH, 'data/ffaw3360.dat'),
          os.path.join(PATH, 'data/ffaw3480.dat'),
          os.path.join(PATH, 'data/cylinder.dat')]:

    d.base_airfoils.append(np.loadtxt(f))

d.update()

# --- 3 ---

# read the blade structure including the geo3d file
st3d = read_bladestructure(os.path.join(PATH, 'data_version_2/Param2_10MW'))

# instantiate class with st3d dict and set additional surface params
st = ComputeDPsParam2(st3d,
                      x=pf['x'],
                      y=pf['y'],
                      z=pf['z'],
                      surface=d.surface)

st.compute()

# save the DPs to new st file
st3d['DPs'] = st.DPs
write_bladestructure(st3d, 'param2_st')

# --- 4 ---

# simple plots
st.plot(coordsys='rotor')
st.plot_topview(coordsys='rotor')

# --- 5 ---

# set the struct_angle to 15 deg
# and shift the main laminates and webs forward

st.struct_angle = 15.
st.cap_center_ss = np.linspace(0.5, 0.0, 20) / 86.366
st.cap_center_ps = np.linspace(0.5, 0.0, 20) / 86.366
st.w02pos += np.linspace(0.5, 0.0, 20) / 86.366
st.w03pos += np.linspace(0.5, 0.0, 20) / 86.366
st.compute()

st.plot(coordsys='rotor', ifig=3, isec=12)

# --- 6 ---
