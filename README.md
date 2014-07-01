# call_MCMAC

## Files 
* elGordo_script - bash script for calling parallel_input_elGordo.py and
  specifies how many cores to use 
* parallel_input_elGordo.py - allows setting random seed according to the
  core number 
* regress_test.py - compares the outputs from the new version of the script those with the old version, both versions have to use the same random seed in order for a fair comparison 

## regression test
the test data for the regression test have the following inputs:

* N_sample = 2500  
* N_bins = 100  
* del_mesh = 100  
* TSM_mesh = 200  
* mock_d_proj_norm_compressed.h5

