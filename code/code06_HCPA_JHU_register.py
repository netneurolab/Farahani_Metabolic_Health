"""

HCP-A

This script transfers the JHU tracts into each subject’s native space.
The transferred tracts are then used to extract FA and MD values in each WM tract.

Note: In the HCP processing pipeline, FA and MD maps are only available in subject (native) space.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import pandas as pd
from globals import path_info_sub, path_T1_native
from globals import path_registration_files, path_native_jhu_files, path_data

#------------------------------------------------------------------------------
# Load the subject IDs
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
subject_ids = df.src_subject_id

# Convert JHU atlas to Subject space
input_JHU_file = path_data + '/JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz'

for s, subid in enumerate(subject_ids):
    os.system('cd /home/afarahani/freesurfer')
    ref_file = path_registration_files + subid + '_V1_MR/MNINonLinear/xfms/standard2acpc_dc.nii.gz'
    native_space = path_T1_native + subid + '_V1_MR/T1w_acpc_dc_restore_1.50.nii.gz'
    command = 'applywarp --in=' + input_JHU_file +\
        ' --out=' + path_native_jhu_files + 'jhu_' + subid +'.nii.gz -r '  + native_space + \
            ' --warp=' + ref_file + ' --interp=nn'
    os.system(command)

    print(s) # Show progress - what is the subject number under conversion?

#------------------------------------------------------------------------------
# End