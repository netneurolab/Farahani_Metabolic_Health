"""

HCP-A

Combine all MRI data

The order of the features is:

    WM- [9 features]
    data_perfsuion,
    data_arrival,
    data_thickness,
    data_myelin,
    FA,
    MD,
    FC,
    SC

Note: The saved data array is named "data_merged.npy".

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from globals import path_results, path_info_sub, path_FC, path_SC
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Load data (vertex-wise) - ASL, structural, and diffusion MRI
#------------------------------------------------------------------------------

perfsuion = np.load(path_results + 'perfusion_all_vertex.npy') # perfusion data (CBF)
arrival = np.load(path_results   + 'arrival_all_vertex.npy')   # perfusion data (ATT)
thickness = np.load(path_results + 'thickness_all_vertex.npy') # structural data (T1w and T2w)
myelin = np.load(path_results    + 'myelin_all_vertex.npy')    # structural data (T1w and T2w)

num_vertices = thickness.shape[0]
num_subjects = thickness.shape[1]

# Parcellate data based on Schaefer-400 parcellation
data_perfsuion = convert_cifti_to_parcellated_SchaeferTian(perfsuion.T,
                                                 'cortex',
                                                 'S1',
                                                 path_results,
                                                 'Y')

data_arrival = convert_cifti_to_parcellated_SchaeferTian(arrival.T,
                                                 'cortex',
                                                 'S1',
                                                 path_results,
                                                 'Y')

data_thickness = convert_cifti_to_parcellated_SchaeferTian(thickness.T,
                                                 'cortex',
                                                 'S1',
                                                 path_results,
                                                 'Y')

data_myelin = convert_cifti_to_parcellated_SchaeferTian(myelin.T,
                                                 'cortex',
                                                 'S1',
                                                 path_results,
                                                 'Y')

data_FAs = np.load(path_results + 'FAs.npy') # diffusion data (FA)
data_MDs = np.load(path_results + 'MDs.npy') # diffusion data (MD)
data_WMH = np.load(path_results + 'WMH.npy') # White matter hyperintensity data

#------------------------------------------------------------------------------
# Combine them into a single matrix
#------------------------------------------------------------------------------

data_merged = np.concatenate((data_WMH,
                              data_perfsuion,
                              data_arrival,
                              data_thickness,
                              data_myelin,
                              data_FAs,
                              data_MDs
                              ), axis = 0)

#------------------------------------------------------------------------------
# Load FC
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
subject_ids = df.src_subject_id
nnodes = 400
FC = np.zeros((num_subjects, nnodes, nnodes))

for s, subid in enumerate(subject_ids):
    s_FC = np.load(path_FC + 'FC_' + subid + '.npy')
    FC[s,:,:] = s_FC
FC = np.abs(FC)
FC_str = np.sum(FC, axis = 1)

data_merged = np.concatenate((data_merged,
                              FC_str.T
                              ), axis = 0) # add FC to the merged data array

#------------------------------------------------------------------------------
# Load SC
#------------------------------------------------------------------------------

nnodes = 400
SC = np.zeros((num_subjects, nnodes, nnodes))

for s, subid in enumerate(subject_ids):
    s_SC = np.load(path_SC + subid + '_sc_3_waytotal-log_parc.npy')
    SC[s,:,:] = s_SC
SC_str = np.sum(SC, axis = 1)

data_merged = np.concatenate((data_merged,
                              SC_str.T
                              ), axis = 0) # add SC to the merged data array

#------------------------------------------------------------------------------
# Save the data for later
#------------------------------------------------------------------------------

np.save(path_results + 'data_merged.npy', data_merged)

#------------------------------------------------------------------------------
# END