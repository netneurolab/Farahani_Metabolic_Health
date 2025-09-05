"""

HCP-A

Combine data of WM tracts - JHU atlas

The feature order is:

    perfusion
    arrival
    FAs
    MDs

Note: The saved data array is named "data_merged_JHU.npy".

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
from globals import path_results

#------------------------------------------------------------------------------

perfusion = np.load(path_results + 'perfusion_jhu.npy')
arrival = np.load(path_results + 'arrival_jhu.npy')
FA = np.load(path_results + 'FAs_jhu.npy')
MD = np.load(path_results + 'MDs_jhu.npy')

data_merged = np.concatenate((perfusion,
                              arrival,
                              FA,
                              MD,
                              ), axis = 0)

np.save(path_results + 'data_merged_JHU.npy', data_merged)

#------------------------------------------------------------------------------
# END