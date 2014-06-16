#!/opt/local/bin/python
"""
Let me try to compare the files without using shell commands
"""

import os
import cPickle
import numpy as np

basepath = "./base_case/"
newpath = "./added_indices/"
baseprefix = "EG_"
newprefix = "EGindices_"
FileNo = 2

var = ['m_1',
       'm_2',
       'z_1',
       'z_2',
       'd_proj',
       'v_rad_obs',
       'v_3d_obs',
       'alpha',
       'd_max',
       'TSM_0',
       'TSM_1',
       'T',
       'prob']

basefiles = [basepath + baseprefix + str(no) + "_" + v + ".pickle" 
        for no in range(FileNo) for v in var]

testfiles = [newpath + newprefix + str(no) + "_" + v + ".pickle" 
        for no in range(FileNo) for v in var]

def compare_pkl(file1, file2):
    f1 = cPickle.load(open(file1, "r"))
    f2 = cPickle.load(open(file2, "r"))
    assert np.array_equal(f1, f2), "{0} and {1} not equal".format(file1,
            file2)

map(compare_pkl, basefiles, testfiles)    
