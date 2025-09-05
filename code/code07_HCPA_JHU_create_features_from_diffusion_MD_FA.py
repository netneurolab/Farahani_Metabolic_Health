"""

HCP-A

Parcellate and save white matter tract FA and MD per individual subject

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import nibabel as nib
from globals import path_native_jhu_files, path_results
from globals import path_info_sub, path_FA_native, path_MD_native, path_data_MD, path_data_FA

#------------------------------------------------------------------------------
# Parcellate individuals' WM FA and MD data
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
subject_ids = df.src_subject_id

nnodes = 20 # number of WM tracts
nsub = len(subject_ids)

FAs = np.zeros((nnodes, nsub))
MDs = np.zeros((nnodes, nsub))

for s, subid in enumerate(subject_ids):
    print(s) # show progress
    jhu_subject = nib.load(path_native_jhu_files + 'jhu_' + subid + '.nii.gz').get_fdata()
    FA_subject = nib.load(path_FA_native +  subid + '_V1_MR/dti_FA.nii.gz').get_fdata()
    MD_subject = nib.load(path_MD_native +  subid + '_V1_MR/dti_MD.nii.gz').get_fdata()

    FA_subject[FA_subject == 0] = np.nan
    MD_subject[MD_subject == 0] = np.nan

    for j in range(nnodes):
        FAs[j, s] = np.nanmean(FA_subject[jhu_subject == j + 1])
        MDs[j, s] = np.nanmean(MD_subject[jhu_subject == j + 1])

    FA_sub = FAs[:, s]
    MD_sub = MDs[:, s]
    np.save(path_data_FA  + subid + '_FA_jhu.npy', FA_sub)
    np.save(path_data_MD  + subid + '_MD_jhu.npy', MD_sub)

np.save(path_results + 'FAs_jhu.npy', FAs)
np.save(path_results + 'MDs_jhu.npy', MDs)

#------------------------------------------------------------------------------
# END