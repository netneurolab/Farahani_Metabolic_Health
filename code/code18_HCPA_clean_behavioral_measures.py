"""

HCP-A

Combine and clean all behavioral measures available for HCP-Aging dataset.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from globals import path_measures, path_info_sub, path_results

#------------------------------------------------------------------------------
# Needed function
#------------------------------------------------------------------------------

def remove_column(df):
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isna().mean() * 100
    
    # Identify columns where the percentage of missing values is greater than 30%
    columns_to_drop = missing_percentage[missing_percentage > 30].index
    
    # Drop these columns from the DataFrame
    df_cleaned = df.drop(columns = columns_to_drop)
    return df_cleaned

#------------------------------------------------------------------------------
# dataset 1
name_measue = 'asr01.txt'
df_asr01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')

df_asr01 = df_asr01.drop(columns = ['collection_id',
                                  'asr01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'comqother',
                                  'version_form',
                                  'interview_age',
                                  'sex'])
#------------------------------------------------------------------------------
# dataset 2
name_measue = 'batbil01.txt'
df_batbil01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_batbil01 = df_batbil01.drop(columns = ['collection_id',
                                  'batbil01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])

combined_df = pd.merge(df_batbil01, df_asr01,
                       on = 'subjectkey', how = 'outer')
#------------------------------------------------------------------------------
# dataset 4
name_measue = 'cogcomp01.txt'
df_cogcomp01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_cogcomp01 = df_cogcomp01.drop(columns = ['collection_id',
                                  'cogcomp01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'interview_language',
                                  'comments1',
                                  'comments2',
                                  'comments3',
                                  'comments4',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_cogcomp01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 5
name_measue = 'dccs01.txt'
df_dccs01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_dccs01 = df_dccs01.drop(columns = ['collection_id',
                                  'dccs01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'interview_language',
                                  'fneproc',
                                  'wcst_ni',
                                  'version_form',
                                  'comqother',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_dccs01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 6
name_measue = 'deldisk01.txt'
df_deldisk01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_deldisk01 = df_deldisk01.drop(columns = ['collection_id',
                                  'deldisk01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'version_form',
                                  'comqother',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'ddisc_valid'])
combined_df = pd.merge(combined_df, df_deldisk01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 7
name_measue = 'edinburgh_hand01.txt'
df_edinburgh_hand01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_edinburgh_hand01 = df_edinburgh_hand01.drop(columns = ['collection_id',
                                  'edinburgh_hand01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_edinburgh_hand01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 8
name_measue = 'er4001.txt'
df_er4001 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_er4001 = df_er4001.drop(columns = ['collection_id',
                                  'er4001_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'comqother'])
combined_df = pd.merge(combined_df, df_er4001,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 9
name_measue = 'facename01.txt'
df_facename01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_facename01 = df_facename01.drop(columns = ['collection_id',
                                  'facename01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'version_form'])
combined_df = pd.merge(combined_df, df_facename01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 10
name_measue = 'flanker01.txt'
df_flanker01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_flanker01 = df_flanker01.drop(columns = ['collection_id',
                                  'flanker01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'interview_language',
                                  'fneproc',
                                  'wcst_ni',
                                  'version_form',
                                  'comqother',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_flanker01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 11
name_measue = 'gales01.txt'
df_gales01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_gales01 = df_gales01.drop(columns = ['collection_id',
                                  'gales01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_gales01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 12
name_measue = 'ipaq01.txt'
df_ipaq01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_ipaq01 = df_ipaq01.drop(columns = ['collection_id',
                                  'ipaq01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_ipaq01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 13
name_measue = 'lbadl01.txt'
df_lbadl01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_lbadl01 = df_lbadl01.drop(columns = ['collection_id',
                                  'lbadl01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_lbadl01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 14
name_measue = 'leap01.txt'
df_leap01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_leap01 = df_leap01.drop(columns = ['collection_id',
                                  'leap01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'comqother',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_leap01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 15
name_measue = 'lswmt01.txt'
df_lswmt01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_lswmt01 = df_lswmt01.drop(columns = ['collection_id',
                                  'lswmt01_id',
                                  'dataset_id',
                                  'fneproc',
                                  'interview_date',
                                  'comqother',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'version_form'])
combined_df = pd.merge(combined_df, df_lswmt01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 16
name_measue = 'mchq01.txt'
df_mchq01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_mchq01 = df_mchq01.drop(columns = ['collection_id',
                                  'mchq01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_mchq01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 17
name_measue = 'medh01.txt'
df_medh01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_medh01 = df_medh01.drop(columns = ['collection_id',
                                  'medh01_id',
                                  'dataset_id',
                                  'comqother',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_medh01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 18
name_measue = 'mendt01.txt'
df_mendt01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_mendt01 = df_mendt01.drop(columns = ['collection_id',
                                  'mendt01_id',
                                  'dataset_id',
                                  'comqother',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_mendt01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 19
name_measue = 'moca01.txt'
df_moca01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_moca01 = df_moca01.drop(columns = ['collection_id',
                                  'moca01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_moca01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 20
name_measue = 'ndar_subject01.txt'
df_ndar_subject01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_ndar_subject01 = df_ndar_subject01.drop(columns = ['collection_id',
                                  'ndar_subject01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'visit',
                                  'site',
                                  'phenotype',
                                  'phenotype_description',
                                  'twins_study',
                                  'sibling_study',
                                  'family_study'])
combined_df = pd.merge(combined_df, df_ndar_subject01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 21
name_measue = 'nffi01.txt'
df_nffi01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_nffi01 = df_nffi01.drop(columns = ['collection_id',
                                  'nffi01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_nffi01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 22
name_measue = 'orrt01.txt'
df_orrt01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_orrt01 = df_orrt01.drop(columns = ['collection_id',
                                  'orrt01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'version_form',
                                  'fneproc',
                                  'wcst_ni',
                                  'version_form',
                                  'comqother',
                                  'primary_language'])
combined_df = pd.merge(combined_df, df_orrt01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 23
name_measue = 'pcps01.txt'
df_pcps01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_pcps01 = df_pcps01.drop(columns = ['collection_id',
                                  'pcps01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'version_form',
                                  'fneproc',
                                  'wcst_ni',
                                  'version_form',
                                  'comqother',
                                  'interview_language'])
combined_df = pd.merge(combined_df, df_pcps01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 24
# X
#------------------------------------------------------------------------------
# dataset 25
def prefix_columns(df, prefix):
    return df.rename(columns={col: f"{prefix}_{col}" for col in df.columns if col != 'subjectkey'})

name_measue = 'preda01.txt'
df_prang01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')

# Create a unique column name for each version
if 'version_form' in df_prang01.columns:
    versions = df_prang01['version_form'].unique()
    for version in versions:
        subset = df_prang01[df_prang01['version_form'] == version]
        for col in subset.columns:
            if col not in ['subjectkey', 'version_form']:
                subset = subset.rename(columns = {col: f"{col}_v{version}"})
        if 'combined_df' in locals():
            combined_df = pd.merge(combined_df, subset.drop(columns = ['version_form']),
                                   on = 'subjectkey', how = 'outer')
        else:
            combined_df = subset.drop(columns = ['version_form'])
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 26
name_measue = 'predd01.txt'
df_predd01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_predd01 = df_predd01.drop(columns = ['collection_id',
                                  'predd01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'fneproc',
                                  'version_form',
                                  'comqother',
                                  'interview_language'])
combined_df = pd.merge(combined_df, df_predd01,
                       on = 'subjectkey',  how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 27
name_measue = 'promisgl01.txt'
df_promisgl01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_promisgl01 = df_promisgl01.drop(columns = ['collection_id',
                                  'promisgl01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'fneproc',
                                  'version_form',
                                  'comqother',
                                  'interview_language'])
combined_df = pd.merge(combined_df, df_promisgl01,
                       on = 'subjectkey',  how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 28
name_measue = 'prsi01.txt'
df_prsi01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_prsi01 = df_prsi01.drop(columns = ['collection_id',
                                  'prsi01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'version_form',
                                  'comqother',
                                  'interview_language'])
combined_df = pd.merge(combined_df, df_prsi01,
                       on = 'subjectkey',  how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 29
name_measue = 'psm01.txt'
df_psm01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_psm01 = df_psm01.drop(columns = ['collection_id',
                                  'psm01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'version_form',
                                  'comqother',
                                  'interview_language'])
combined_df = pd.merge(combined_df, df_psm01,
                       on = 'subjectkey',  how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 30
name_measue = 'psqi01.txt'
df_psqi01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_psqi01 = df_psqi01.drop(columns = ['collection_id',
                                  'psqi01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'comqother'])
combined_df = pd.merge(combined_df, df_psqi01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 31
name_measue = 'pss01.txt'
df_pss01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_pss01 = df_pss01.drop(columns = ['collection_id',
                                  'pss01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'version_form',
                                  'respondent',
                                  'interview_language'])
df_pss01 = prefix_columns(df_pss01, name_measue.replace('.txt', ''))
combined_df = pd.merge(combined_df, df_pss01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 32
name_measue = 'ravlt01.txt'
df_ravlt01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_ravlt01 = df_ravlt01.drop(columns = ['collection_id',
                                  'ravlt01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_ravlt01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 33
name_measue = 'self_effic01.txt'
df_self_effic01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_self_effic01 = df_self_effic01.drop(columns = ['collection_id',
                                  'self_effic01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'version_form',
                                  'interview_language'])
df_self_effic01 = prefix_columns(df_self_effic01, name_measue.replace('.txt', ''))
combined_df = pd.merge(combined_df, df_self_effic01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 34
name_measue = 'ssaga_cover_demo01.txt'
df_ssaga_cover_demo01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_ssaga_cover_demo01 = df_ssaga_cover_demo01.drop(columns = ['collection_id',
                                  'ssaga_cover_demo01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
combined_df = pd.merge(combined_df, df_ssaga_cover_demo01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 35

name_measue = 'tlbx_emsup01.txt'
df_tlbx_emsup01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tlbx_emsup01 = df_tlbx_emsup01.drop(columns = ['collection_id',
                                  'tlbx_emsup01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'interview_language',
                                  'primary_language'])
df_tlbx_emsup01 = prefix_columns(df_tlbx_emsup01, name_measue.replace('.txt', ''))
# Create a unique column name for each version
if 'tlbx_emsup01_version_form' in df_tlbx_emsup01.columns:
    versions = df_tlbx_emsup01['tlbx_emsup01_version_form'].unique()
    for version in versions:
        subset = df_tlbx_emsup01[df_tlbx_emsup01['tlbx_emsup01_version_form'] == version]
        for col in subset.columns:
            if col not in ['subjectkey', 'tlbx_emsup01_version_form']:
                subset = subset.rename(columns = {col: f"{col}_v{version}"})
        if 'combined_df' in locals():
            combined_df = pd.merge(combined_df, subset.drop(columns = ['tlbx_emsup01_version_form']),
                                   on = 'subjectkey', how = 'outer')
        else:
            combined_df = subset.drop(columns = ['tlbx_emsup01_version_form'])

print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 37
name_measue = 'tlbx_friend01.txt'
df_tlbx_friend01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tlbx_friend01 = df_tlbx_friend01.drop(columns = ['collection_id',
                                  'tlbx_friend01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'version_form',
                                  'interview_language'])
df_tlbx_friend01 = prefix_columns(df_tlbx_friend01, name_measue.replace('.txt', ''))
combined_df = pd.merge(combined_df, df_tlbx_friend01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 38
name_measue = 'tlbx_motor01.txt'
df_tlbx_motor01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tlbx_motor01 = df_tlbx_motor01.drop(columns = ['collection_id',
                                  'tlbx_motor01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'interview_language'])
df_tlbx_motor01 = prefix_columns(df_tlbx_motor01, name_measue.replace('.txt', ''))
if 'tlbx_motor01_version_form' in df_tlbx_motor01.columns:
    versions = df_tlbx_motor01['tlbx_motor01_version_form'].unique()
    for version in versions:
        subset = df_tlbx_motor01[df_tlbx_motor01['tlbx_motor01_version_form'] == version]
        for col in subset.columns:
            if col not in ['subjectkey', 'tlbx_motor01_version_form']:
                subset = subset.rename(columns = {col: f"{col}_v{version}"})
        if 'combined_df' in locals():
            combined_df = pd.merge(combined_df, subset.drop(columns = ['tlbx_motor01_version_form']),
                                   on = 'subjectkey', how = 'outer')
        else:
            combined_df = subset.drop(columns = ['tlbx_motor01_version_form'])
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 39
name_measue = 'tlbx_perhost01.txt'
df_tlbx_perhost01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tlbx_perhost01 = df_tlbx_perhost01.drop(columns = ['collection_id',
                                  'tlbx_perhost01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'version_form',
                                  'interview_language'])
df_tlbx_perhost01 = prefix_columns(df_tlbx_perhost01, name_measue.replace('.txt', ''))
combined_df = pd.merge(combined_df, df_tlbx_perhost01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 40
name_measue = 'tlbx_rej01.txt'
df_tlbx_rej01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tlbx_rej01 = df_tlbx_rej01.drop(columns = ['collection_id',
                                  'tlbx_rej01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'version_form',
                                  'interview_language'])
df_tlbx_rej01 = prefix_columns(df_tlbx_rej01, name_measue.replace('.txt', ''))
combined_df = pd.merge(combined_df, df_tlbx_rej01,
                       on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 41
name_measue = 'tlbx_sensation01.txt'
df_tlbx_sensation01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tlbx_sensation01 = df_tlbx_sensation01.drop(columns = ['collection_id',
                                  'tlbx_sensation01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'interview_language'])
df_tlbx_sensation01 = prefix_columns(df_tlbx_sensation01, name_measue.replace('.txt', ''))
if 'tlbx_sensation01_version_form' in df_tlbx_sensation01.columns:
    versions = df_tlbx_sensation01['tlbx_sensation01_version_form'].unique()
    for version in versions:
        subset = df_tlbx_sensation01[df_tlbx_sensation01['tlbx_sensation01_version_form'] == version]
        for col in subset.columns:
            if col not in ['subjectkey', 'tlbx_sensation01_version_form']:
                subset = subset.rename(columns = {col: f"{col}_v{version}"})
        if 'combined_df' in locals():
            combined_df = pd.merge(combined_df, subset.drop(columns = ['tlbx_sensation01_version_form']),
                                   on = 'subjectkey', how = 'outer')
        else:
            combined_df = subset.drop(columns = ['tlbx_sensation01_version_form'])
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 42
name_measue = 'tlbx_wellbeing01.txt'
df_tlbx_wellbeing01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tlbx_wellbeing01 = df_tlbx_wellbeing01.drop(columns = ['collection_id',
                                  'tlbx_wellbeing01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'interview_language',
                                  'primary_language'])
df_tlbx_wellbeing01 = prefix_columns(df_tlbx_wellbeing01, name_measue.replace('.txt', ''))
if 'tlbx_wellbeing01_version_form' in df_tlbx_wellbeing01.columns:
    versions = df_tlbx_wellbeing01['tlbx_wellbeing01_version_form'].unique()
    for version in versions:
        subset = df_tlbx_wellbeing01[df_tlbx_wellbeing01['tlbx_wellbeing01_version_form'] == version]
        for col in subset.columns:
            if col not in ['subjectkey', 'tlbx_wellbeing01_version_form']:
                subset = subset.rename(columns = {col: f"{col}_v{version}"})
        if 'combined_df' in locals():
            combined_df = pd.merge(combined_df, subset.drop(columns = ['tlbx_wellbeing01_version_form']),
                                   on = 'subjectkey', how = 'outer')
        else:
            combined_df = subset.drop(columns = ['tlbx_wellbeing01_version_form'])
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 43
name_measue = 'tpvt01.txt'
df_tpvt01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_tpvt01 = df_tpvt01.drop(columns = ['collection_id',
                                  'tpvt01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'wcst_ni',
                                  'fneproc',
                                  'comqother',
                                  'version_form',
                                  'interview_language'])
combined_df = pd.merge(combined_df, df_tpvt01, on = 'subjectkey', how = 'outer')
print(len(combined_df))
#------------------------------------------------------------------------------
# dataset 44
name_measue = 'trail_ca01.txt'
df_trail_ca01 = pd.read_csv(path_measures + name_measue, delimiter = '\t')
df_trail_ca01 = df_trail_ca01.drop(columns = ['collection_id',
                                  'trail_ca01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex',
                                  'versionchildadult'])
combined_df = pd.merge(combined_df, df_trail_ca01, on = 'subjectkey', how ='outer')
print(len(combined_df))

#------------------------------------------------------------------------------
# Add more information to the column names
#------------------------------------------------------------------------------

# Extract the first row values
first_row_values = combined_df.iloc[0]
new_column_names = [f"{col} ({val})" for col, val in zip(combined_df.columns, first_row_values)]
combined_df.columns = new_column_names
combined_df = combined_df.drop(0).reset_index(drop=True)
print(combined_df)
combined_df = combined_df.rename(columns = {'subjectkey (The NDAR Global Unique Identifier (GUID) for research subject)': 'subjectkey'})

#------------------------------------------------------------------------------

# load subject names
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')

# Keep only the three desired columns
columns_to_keep = ['subjectkey', 'sex', 'interview_age']
df = df[columns_to_keep]
a = pd.merge(df, combined_df, on = 'subjectkey', how = 'left')

# Function to clean DataFrame by removing columns that cannot be converted to numeric (excluding the first row)
def clean_dataframe(df):
    columns_to_drop = []
    for col in df.columns:
        if col == 'subjectkey':
            continue
        series_to_check = df[col].iloc[0:]
        try:
            pd.to_numeric(series_to_check, errors='raise')
        except ValueError:
            columns_to_drop.append(col)
        except TypeError:
            columns_to_drop.append(col)

        columns_to_drop.append('version_form (Form used/assessment name)')
        columns_to_drop.append('family_user_def_id (Family Pedigree User-Defined ID)')
        columns_to_drop.append('interview_age_vNIH Toolbox Fear-Somatic Arousal FF Age 18+ v2.0 (nan)')
        columns_to_drop.append('interview_age_vNIH Toolbox Fear-Affect CAT Age 18+ v2.0 (nan)')
    # Drop columns that failed conversion
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

cleaned_df = clean_dataframe(a)

print(cleaned_df)
def clean_dataframe(df):
    columns_to_drop = []
    for col in df.columns:
        if col == 'subjectkey':
            continue
        # Calculate the percentage of NaN values excluding the first row
        nan_percentage = df[col].iloc[0:].isna().mean() * 100
        # Check if more than 20% of the column is NaN
        if nan_percentage > 20:
            columns_to_drop.append(col)
            continue  # Skip further checks and move to the next column
        # Calculate the number of unique values in the column
        unique_values = np.unique(df[col].dropna())
        if len(unique_values) < 2:
            columns_to_drop.append(col)
            continue
    # Drop columns that failed conversion or have too many NaNs
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned
finalDF = clean_dataframe(cleaned_df)

for col in finalDF.columns:
    if col == 'subjectkey':
        continue
    else:
    # Check if the column contains only binary values (0 and 1)
        unique_values = np.unique(finalDF[col].dropna())
        print(col)
        print(unique_values)
        
        # Convert the unique values to float
        unique_values = unique_values.astype(float)
        
        if sum(unique_values[:2]) < 3.5:
            finalDF = finalDF.drop(columns=col)

finalDF.to_csv(path_results + 'no_plasma_finalDF_all.csv', index = False)
#------------------------------------------------------------------------------
# END