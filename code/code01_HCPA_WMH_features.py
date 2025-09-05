"""

HCP-A

Get the WMH data for HCP-Aging dataset.
140 out of 678 have missing values for WMH data.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from globals import path_WMH_hcp, path_results, path_info_sub

#------------------------------------------------------------------------------

# Who are the subjects that have the ASL data in HCP-aging?
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
subject_ids = df.src_subject_id.tolist()
nsub = len(subject_ids)

# load the WMH data - for subjects who have the data processed
wmh_data = pd.read_csv(path_WMH_hcp + 'Info_WMH_hcpa.csv')
subjects_wmh = wmh_data['hcp_csv'].astype(str).str[-7:]

# Ensure WMH values are numeric
wmh_values = wmh_data.drop(columns = ['hcp_csv']).apply(pd.to_numeric, errors = 'coerce').to_numpy().T  # shape: (n_features, n_subjects)

# Filter the WMH data to see which subject have both WMH and ASL data available
WMHs = np.full((wmh_values.shape[0], nsub), np.nan)
for i, sub in enumerate(subject_ids):
    sub_num = str(sub)[-7:]  # Last 7 digits of subject ID
    matches = np.where(subjects_wmh == sub_num)[0]
    if len(matches) > 0:
        WMHs[:, i] = wmh_values[:, matches[0]]
    else:
        print(sub)

# Save WMH array
np.save(path_results + 'WMH.npy', WMHs)

# Count and print number of subjects with missing WMH data (out of the 678 subjects)
missing_subjects = np.sum(np.all(np.isnan(WMHs), axis = 0))
print(f'Number of subjects with missing WMH data: {missing_subjects}')

#------------------------------------------------------------------------------
# END