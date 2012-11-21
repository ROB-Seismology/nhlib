"""
from hazard.psha.attenuation import *




M = 5.
r = 50.
T = 1.
h = 0
num_sigma=0
force_zero_depth=False
soil_type="rock"
mechanism="normal"
damping=5

gmpe = Ambraseys1996GMPE()
periods, accelerations = gmpe.get_spectrum(M, r, T, h=h, force_zero_depth=force_zero_depth, soil_type=soil_type, mechanism=mechanism, damping=damping)
"""

from nhlib.gsim.ambraseys_et_al_1996 import AmbraseysEtAl1996

gmpe = AmbraseysEtAl1996()