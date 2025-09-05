"""

HCP-A

This script transfers the Schaefer-400 parcellation into each subject’s native space.
The transferred parcellations are then used to extract parcel-wise cortical FA and MD values.

Note: In the HCP processing pipeline, FA and MD maps are only available in subject (native) space.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import pandas as pd
from globals import path_info_sub, path_T1_native
from globals import path_registration_files, path_native_schaefer_files

#------------------------------------------------------------------------------
# Load the subject IDs
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
subject_ids = df.src_subject_id

# Convert Schaefer to Subject space
input_schaefer_file = '/home/afarahani/Desktop/multi_factors/data/schaefer_HCP_MNI/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'

for s, subid in enumerate(subject_ids):
    os.system('cd /home/afarahani/freesurfer')
    ref_file = path_registration_files + subid + '_V1_MR/MNINonLinear/xfms/standard2acpc_dc.nii.gz'
    native_space = path_T1_native + subid + '_V1_MR/T1w_acpc_dc_restore_1.50.nii.gz'
    command = 'applywarp --in=' + input_schaefer_file +\
        ' --out=' + path_native_schaefer_files + 'schaefer_' + subid + '.nii.gz -r '  + native_space + \
            ' --warp=' + ref_file + ' --interp=nn'
    os.system(command)

    print(s) # Show progress - what is the subject number under conversion?

#------------------------------------------------------------------------------
# END