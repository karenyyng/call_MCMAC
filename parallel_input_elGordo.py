'''
Author: Will Dawson
Modified by: Karen Yin-Yee Ng
Comment:
This is the code that gets the fits file from James for the ElGordo weak lensing
mass analysis and massage it so that it can be digested by Will 's MCMAC.py code
parallel version!
'''
from __future__ import division
import sys
from MCMAC import MCengine
import h5py
import re

# this specifies the name of the output file
if len(sys.argv) < 2:
    sys.exit("WARNING: Oops you need to specify what prefix you want \
             to use.Type $parallel_inputElgordo.py PREFIX \
             instead. Aborting.")
else:
    prefix = sys.argv[1]
    found = re.search("[a-zA-Z_]([0-9]+)", prefix)
    try:
        seed = int(found.group(1))
    except AttributeError:
        sys.exit("prefix should have the form of [a-zA-Z_]([0-9]+)")
print "file prefix is ", prefix
print "seed is {0}".format(seed)

# inputs for the TSM module
# N_sample is the number of iterations of the MC runs
prefix = sys.argv[1]
N_sample = 2500
N_bins = 100
del_mesh = 100
TSM_mesh = 200


f = h5py.File("mock_d_proj_norm_compressed.h5", "r")
m_main = f["m_main"][...]
m_sub = f["m_sub"][...]
z_main = f["z_main"][...]
z_sub = f["z_sub"][...]
D_proj = f["D_proj"][...]
f.close()

###
# Perform the TSM calculation
###
MCengine(N_sample, m_main, m_sub, z_main, z_sub, D_proj, prefix,
         del_mesh=del_mesh, TSM_mesh=TSM_mesh, seed=seed)
