"""

UK biobank

The script generates a CSV file named path_ukbiobank + "merged_biodata_with_cbf.csv".
This CSV includes biomarker data and ASL-derived measures (CBF and ATT).
We exclude participants with reported mental-health or neurological conditions.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import re
import pandas as pd
from globals import path_ukbiobank

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

main_data = pd.read_csv(path_ukbiobank + 'data_082025_withimaging.csv') # raw data - UK Biobank

#------------------------------------------------------------------------------
# Remove subjects with mental health problems
#------------------------------------------------------------------------------

mental_health_columns = [col for col in main_data.columns
                         if col.startswith('mental_health_problems_ever_diagnosed_by_a_professional_20544-0.')]

diseases_mental_to_exclude = [
    'Depression',
    'Mania, hypomania, bipolar or manic-depression',
    "Autism, Asperger's or autistic spectrum disorder",
    'Panic attacks',
    'Obsessive compulsive disorder (OCD)',
    'Schizophrenia',
    'A personality disorder',
    'Attention deficit or attention deficit and hyperactivity disorder (ADD/ADHD)'
    ]

# Loop through relevant columns and mark rows to exclude
exclude_mask = pd.Series(False, index = main_data.index)
for col in mental_health_columns:
    col_values = main_data[col].astype(str)
    for disease in diseases_mental_to_exclude:
        exclude_mask |= col_values.str.contains(disease, na = False)
main_data = main_data.loc[~exclude_mask].copy()
main_data.reset_index(drop=True, inplace=True)

#------------------------------------------------------------------------------
# Remove subjects with physical health problems
#------------------------------------------------------------------------------

physical_health_columns = [col for col in main_data.columns
                           if col.startswith('non-cancer_illness_code_self-reported_20002-0.')]

diseases_physical_to_exclude = [
     'parkinsons disease',
     'dementia/alzheimers/cognitive impairment',
     'chronic/degenerative neurological problem',
     'acute infective polyneuritis/guillain-barre syndrome',
     'multiple sclerosis',
     'other demyelinating disease (not multiple sclerosis)',
     'stroke',
     'brain haemorrhage',
     'brain abscess/intracranial abscess',
     'cerebral aneurysm',
     'cerebral palsy',
     'encephalitis',
     'epilepsy',
     'head injury',
     'infection of nervous system', 'ischaemic stroke',
     'meningioma / benign meningeal tumour',
     'meningitis',
     'motor neurone disease',
     'neurological injury/trauma',
     'spina bifida',
     'subdural haemorrhage/haematoma',
     'subarachnoid haemorrhage',
     'transient ischaemic attack (tia)'
    ]

# Loop through relevant columns and mark rows to exclude
exclude_mask = pd.Series(False, index = main_data.index)
for col in physical_health_columns:
    col_values = main_data[col].astype(str)
    for disease in diseases_physical_to_exclude:
        exclude_mask |= col_values.str.contains(disease, na = False)
main_data = main_data.loc[~exclude_mask].copy()
main_data.reset_index(drop=True, inplace=True)

#------------------------------------------------------------------------------
# Keep the colums that are of interest in this study
#------------------------------------------------------------------------------

interesting_labels = [
    'eid',
    'sex_31-0.0',
    'waist_circumference_48-2.0',
    'hip_circumference_49-2.0',
    'visceral_adipose_tissue_volume_vat_22407-2.0',
    'body_fat_percentage_23099-2.0',
    'basal_metabolic_rate_23105-2.0',
    'total_fat_mass_23278-2.0',
    'total_fat-free_mass_23279-2.0',
    'total_lean_mass_23280-2.0',
    'total_tissue_fat_percentage_23281-2.0',
    'total_mass_23283-2.0',
    'vat_visceral_adipose_tissue_mass_23288-2.0',
    'vat_visceral_adipose_tissue_volume_23289-2.0',
    'potassium_in_urine_30520-0.0',
    'sodium_in_urine_30530-0.0',
    'alkaline_phosphatase_30610-0.0',
    'alanine_aminotransferase_30620-0.0',
    'aspartate_aminotransferase_30650-0.0',
    'urea_30670-0.0',
    'calcium_30680-0.0',
    'cholesterol_30690-0.0',
    'creatinine_30700-0.0',
    'c-reactive_protein_30710-0.0',
    'glucose_30740-0.0',
    'glycated_haemoglobin_hba1c_30750-0.0',
    'hdl_cholesterol_30760-0.0',
    'ldl_direct_30780-0.0',
    'oestradiol_30800-0.0',
    'total_bilirubin_30840-0.0',
    'testosterone_30850-0.0',
    'total_protein_30860-0.0',
    'triglycerides_30870-0.0',
    'vitamin_d_30890-0.0',
    'body_mass_index_bmi_21001-2.0',
    'diastolic_blood_pressure_automated_reading_4079-0.1',
    'systolic_blood_pressure_automated_reading_4080-0.1',
    'age_when_attended_assessment_centre_21003-0.0',
    'age_when_attended_assessment_centre_21003-2.0',
    'body_fat_percentage_23099-0.0',
    'waist_circumference_48-0.0',
    'hip_circumference_49-0.0',
    'body_mass_index_bmi_21001-0.0',
    'diastolic_blood_pressure_automated_reading_4079-0.0',
    'systolic_blood_pressure_automated_reading_4080-0.0',
    ]

#------------------------------------------------------------------------------
# Area, volume and mean thickness of cortical regions
#------------------------------------------------------------------------------

cortical_names = [
    'caudalanteriorcingulate',
    'caudalmiddlefrontal',
    'cuneus',
    'entorhinal',
    'fusiform',
    'inferiorparietal',
    'inferiortemporal',
    'insula',
    'isthmuscingulate',
    'lateraloccipital',
    'lateralorbitofrontal',
    'lingual',
    'medialorbitofrontal',
    'middletemporal',
    'paracentral',
    'parahippocampal',
    'parsopercularis',
    'parsorbitalis',
    'parstriangularis',
    'pericalcarine',
    'postcentral',
    'posteriorcingulate',
    'precentral',
    'precuneus',
    'rostralanteriorcingulate',
    'rostralmiddlefrontal',
    'superiorfrontal',
    'superiorparietal',
    'superiortemporal',
    'supramarginal',
    'transversetemporal',
    ]

ordered_cols_final = []
ordered_cols_final.extend(interesting_labels)

for feature_prefix in ['area_of_', 'volume_of_', 'mean_thickness_of_']:
    for region in cortical_names:
        left_regex = re.compile(f"{feature_prefix}{region}_left_hemisphere.*")
        right_regex = re.compile(f"{feature_prefix}{region}_right_hemisphere.*")
        left_matches = [col for col in main_data.columns if left_regex.match(col)]
        right_matches = [col for col in main_data.columns if right_regex.match(col)]
        ordered_cols_final.extend(left_matches)
        ordered_cols_final.extend(right_matches)

#------------------------------------------------------------------------------
# WM tracts: fa, md, isovf, icvf
#------------------------------------------------------------------------------

jhu_names = [
    'anterior_corona_radiata',
    'anterior_limb_of_internal_capsule',
    'cerebral_peduncle',
    'cingulum_cingulate_gyrus',
    'cingulum_hippocampus',
    'corticospinal_tract',
    'external_capsule',
    "fornix_cres+stria_terminalis",
    'inferior_cerebellar_peduncle',
    'medial_lemniscus',
    'posterior_corona_radiata',
    'posterior_limb_of_internal_capsule',
    'posterior_thalamic_radiation',
    'retrolenticular_part_of_internal_capsule',
    'sagittal_stratum',
    'superior_cerebellar_peduncle',
    'superior_corona_radiata',
    'superior_fronto-occipital_fasciculus',
    'superior_longitudinal_fasciculus',
    'tapetum',
    'uncinate_fasciculus',
    ]

JHU_single_names = [
    'body_of_corpus_callosum',
    'fornix',
    'genu_of_corpus_callosum',
    'middle_cerebellar_peduncle',
    'pontine_crossing_tract',
    'splenium_of_corpus_callosum',
    ]

for feature_prefix in ['mean_fa_in_', 'mean_md_in_', 'mean_isovf_in_', 'mean_icvf_in_']:
    for region in jhu_names:
        escaped_region = re.escape(region)
        left_regex = re.compile(f"{feature_prefix}{escaped_region}_on_fa_skeleton_left.*")
        right_regex = re.compile(f"{feature_prefix}{escaped_region}_on_fa_skeleton_right.*")
        left_matches = [col for col in main_data.columns if left_regex.match(col)]
        right_matches = [col for col in main_data.columns if right_regex.match(col)]
        ordered_cols_final.extend(left_matches)
        ordered_cols_final.extend(right_matches)
        
    for region_single in JHU_single_names:
        regex = re.compile(f"{feature_prefix}{region_single}_on_fa_skeleton_*")
        matches = [col for col in main_data.columns if regex.match(col)]
        ordered_cols_final.extend(matches)

#------------------------------------------------------------------------------
# Other volumes of interest
#------------------------------------------------------------------------------

# both right and left hemispheres
subcortical_regions = [
    'accumbens-area',
    'accumbens',
    'thalamus',
    'thalamus-proper',
    'putamen',
    'pallidum',
    'hippocampus',
    'caudate',
    'amygdala',
    'choroid-plexus',
    'cerebralwhitematter',
    'cerebellum-white-matter',
    'cerebellum-cortex',
    'cortex',
    'ventraldc',
    'grey_matter_in_thalamus',
    'grey_matter_in_ventral_striatum',
    'grey_matter_in_putamen',
    'grey_matter_in_pallidum',
    'grey_matter_in_hippocampus',
    'grey_matter_in_caudate',
    'grey_matter_in_amygdala',
    ]

whole_brain_regions = [
    'brainsegnotvent',
    'brainsegnotventsurf',
    'brain-stem',
    'brainseg',
    'totalgray',
    'subcortgray'
    ]

for region in subcortical_regions:
    escaped_region = re.escape(region)
    right_regex = re.compile(f"volume_of_{escaped_region}_right*")
    left_regex = re.compile(f"volume_of_{escaped_region}_left*")

    left_matches = [col for col in main_data.columns if left_regex.match(col)]
    right_matches = [col for col in main_data.columns if right_regex.match(col)]
    ordered_cols_final.extend(left_matches)
    ordered_cols_final.extend(right_matches)

for region_single in whole_brain_regions:
    regex = re.compile(f"volume_of_{region_single}_whole_brain_*")
    matches = [col for col in main_data.columns if regex.match(col)]
    ordered_cols_final.extend(matches)
            
#------------------------------------------------------------------------------
# Vessel-related measures (WMH)
#------------------------------------------------------------------------------

vessel_related = [
    'total_volume_of_white_matter_hyperintensities_from_t1_and_t2_flair_images_25781-2.0',
    'volume_of_wm-hypointensities_whole_brain_26528-2.0',
    ]

ordered_cols_final.extend(vessel_related)

#------------------------------------------------------------------------------
# Add also ASL-related measures
#------------------------------------------------------------------------------

# Remove the unwanted measures and do the column
selected_data = main_data[ordered_cols_final]

# Load the ASL-derived data
cbf_data = pd.read_csv(path_ukbiobank + 'ukb_category_119.csv') # raw ASL data

# Drop rows where all columns except 'eid' are NaN
cbf_data_with_data = cbf_data.dropna(subset = cbf_data.columns.difference(['eid']),
                                     how = 'all')

# Merge selected_data (demographics, blood/urine tests, etc.) with cbf_data_with_data on 'eid'
selected_data_with_cbf = pd.merge(
    selected_data,
    cbf_data_with_data[['eid']], # only need the eids that have ASL-derived data
    on='eid',
    how='inner' # keep rows where eid is present in both
)

merged_data = pd.merge(
    selected_data,
    cbf_data_with_data,
    on='eid',
    how='inner'
)

#------------------------------------------------------------------------------
# Make the namings proper for ASL-derived data
#------------------------------------------------------------------------------

names = merged_data.columns
names_to_change = {
    '24405-2.0': 'Mean ATT in Caudate (left)-2.0',
    '24405-3.0': 'Mean ATT in Caudate (left)-3.0',
    '24404-2.0': 'Mean ATT in Caudate (right)-2.0',
    '24404-3.0': 'Mean ATT in Caudate (right)-3.0',
    '24388-2.0': 'Mean ATT in Cerebellum in grey matter (left)-2.0',
    '24388-3.0': 'Mean ATT in Cerebellum in grey matter (left)-3.0',
    '24387-2.0': 'Mean ATT in Cerebellum in grey matter (right)-2.0',
    '24387-3.0': 'Mean ATT in Cerebellum in grey matter (right)-3.0',
    '24390-2.0': 'Mean ATT in Frontal Lobe in grey matter (left)-2.0',
    '24390-3.0': 'Mean ATT in Frontal Lobe in grey matter (left)-3.0',
    '24389-2.0': 'Mean ATT in Frontal Lobe in grey matter (right)-2.0',
    '24389-3.0': 'Mean ATT in Frontal Lobe in grey matter (right)-3.0',
    '24398-2.0': 'Mean ATT in Internal Carotid Artery vascular territory in grey matter (left)-2.0',
    '24398-3.0': 'Mean ATT in Internal Carotid Artery vascular territory in grey matter (left)-3.0',
    '24397-2.0': 'Mean ATT in Internal Carotid Artery vascular territory in grey matter (right)-2.0',
    '24397-3.0': 'Mean ATT in Internal Carotid Artery vascular territory in grey matter (right)-3.0',
    '24392-2.0': 'Mean ATT in Occipital Lobe in grey matter (left)-2.0',
    '24392-3.0': 'Mean ATT in Occipital Lobe in grey matter (left)-3.0',
    '24391-2.0': 'Mean ATT in Occipital Lobe in grey matter (right)-2.0',
    '24391-3.0': 'Mean ATT in Occipital Lobe in grey matter (right)-3.0',
    '24394-2.0': 'Mean ATT in Parietal Lobe in grey matter (left)-2.0',
    '24394-3.0': 'Mean ATT in Parietal Lobe in grey matter (left)-3.0',
    '24393-2.0': 'Mean ATT in Parietal Lobe in grey matter (right)-2.0',
    '24393-3.0': 'Mean ATT in Parietal Lobe in grey matter (right)-3.0',
    '24407-2.0': 'Mean ATT in Putamen (left)-2.0',
    '24407-3.0': 'Mean ATT in Putamen (left)-3.0',
    '24406-2.0': 'Mean ATT in Putamen (right)-2.0',
    '24406-3.0': 'Mean ATT in Putamen (right)-3.0',
    '24396-2.0': 'Mean ATT in Temporal Lobe in grey matter (left)-2.0',
    '24396-3.0': 'Mean ATT in Temporal Lobe in grey matter (left)-3.0',
    '24395-2.0': 'Mean ATT in Temporal Lobe-2.0',
    '24395-3.0': 'Mean ATT in Temporal Lobe-3.0',
    '24409-2.0': 'Mean ATT in Thalamus (left)-2.0',
    '24409-3.0': 'Mean ATT in Thalamus (left)-3.0',
    '24408-2.0': 'Mean ATT in Thalamus (right)-2.0',
    '24408-3.0': 'Mean ATT in Thalamus (right)-3.0',
    '24399-2.0': 'Mean ATT in VertebroBasilar Arteries vascular territories in grey matter-2.0',
    '24399-3.0': 'Mean ATT in VertebroBasilar Arteries vascular territories in grey matter-3.0',
    '24402-2.0': 'Mean ATT in cerebrum in white matter (left)-2.0',
    '24402-3.0': 'Mean ATT in cerebrum in white matter (left)-3.0',
    '24403-2.0': 'Mean ATT in cerebrum in white matter (right)-2.0',
    '24403-3.0': 'Mean ATT in cerebrum in white matter (right)-3.0',
    '24401-2.0': 'Mean ATT in cerebrum in white matter and >50% cerebral partial volume-2.0',
    '24401-3.0': 'Mean ATT in cerebrum in white matter and >50% cerebral partial volume-3.0',
    '24386-2.0': 'Mean ATT in cortex in grey matter-2.0',
    '24386-3.0': 'Mean ATT in cortex in grey matter-3.0',
    '24385-2.0': 'Mean ATT in whole brain in grey matter-2.0',
    '24385-3.0': 'Mean ATT in whole brain in grey matter-3.0',
    '24400-2.0': 'Mean ATT in whole brain in white matter-2.0',
    '24400-3.0': 'Mean ATT in whole brain in white matter-3.0',
    
    '24380-2.0': 'Mean CBF in Caudate (left)-2.0',
    '24380-3.0': 'Mean CBF in Caudate (left)-3.0',
    '24379-2.0': 'Mean CBF in Caudate (right)-2.0',
    '24379-3.0': 'Mean CBF in Caudate (right)-3.0',
    '24363-2.0': 'Mean CBF in Cerebellum in grey matter (left)-2.0',
    '24363-3.0': 'Mean CBF in Cerebellum in grey matter (left)-3.0',
    '24362-2.0': 'Mean CBF in Cerebellum in grey matter (right)-2.0',
    '24362-3.0': 'Mean CBF in Cerebellum in grey matter (right)-3.0',
    '24365-2.0': 'Mean CBF in Frontal Lobe in grey matter (left)-2.0',
    '24365-3.0': 'Mean CBF in Frontal Lobe in grey matter (left)-3.0',
    '24364-2.0': 'Mean CBF in Frontal Lobe in grey matter (right)-2.0',
    '24364-3.0': 'Mean CBF in Frontal Lobe in grey matter (right)-3.0',
    '24373-2.0': 'Mean CBF in Internal Carotid Artery vascular territory in grey matter (left)-2.0',
    '24373-3.0': 'Mean CBF in Internal Carotid Artery vascular territory in grey matter (left)-3.0',
    '24372-2.0': 'Mean CBF in Internal Carotid Artery vascular territory in grey matter (right)-2.0',
    '24372-3.0': 'Mean CBF in Internal Carotid Artery vascular territory in grey matter (right)-3.0',
    '24367-2.0': 'Mean CBF in Occipital Lobe in grey matter (left)-2.0',
    '24367-3.0': 'Mean CBF in Occipital Lobe in grey matter (left)-3.0',
    '24366-2.0': 'Mean CBF in Occipital Lobe in grey matter (right)-2.0',
    '24366-3.0': 'Mean CBF in Occipital Lobe in grey matter (right)-3.0',
    '24369-2.0': 'Mean CBF in Parietal Lobe in grey matter (left)-2.0',
    '24369-3.0': 'Mean CBF in Parietal Lobe in grey matter (left)-3.0',
    '24368-2.0': 'Mean CBF in Parietal Lobe in grey matter (right)-2.0',
    '24368-3.0': 'Mean CBF in Parietal Lobe in grey matter (right)-3.0',
    '24382-2.0': 'Mean CBF in Putamen (left)-2.0',
    '24382-3.0': 'Mean CBF in Putamen (left)-3.0',
    '24381-2.0': 'Mean CBF in Putamen (right)-2.0',
    '24381-3.0': 'Mean CBF in Putamen (right)-3.0',
    '24371-2.0': 'Mean CBF in Temporal Lobe in grey matter (left)-2.0',
    '24371-3.0': 'Mean CBF in Temporal Lobe in grey matter (left)-3.0',
    '24370-2.0': 'Mean CBF in Temporal Lobe in grey matter (right)-2.0',
    '24370-3.0': 'Mean CBF in Temporal Lobe in grey matter (right)-3.0',
    '24384-2.0': 'Mean CBF in Thalamus (left)-2.0',
    '24384-3.0': 'Mean CBF in Thalamus (left)-3.0',
    '24383-2.0': 'Mean CBF in Thalamus (right)-2.0',
    '24383-3.0': 'Mean CBF in Thalamus (right)-3.0',
    '24374-2.0': 'Mean CBF in VertebroBasilar Arteries vascular territories in grey matter-2.0',
    '24374-3.0': 'Mean CBF in VertebroBasilar Arteries vascular territories in grey matter-3.0',
    '24377-2.0': 'Mean CBF in cerebrum in white matter (left)-2.0',
    '24377-3.0': 'Mean CBF in cerebrum in white matter (left)-3.0',
    '24378-2.0': 'Mean CBF in cerebrum in white matter (right)-2.0',
    '24378-3.0': 'Mean CBF in cerebrum in white matter (right)-3.0',
    '24376-2.0': 'Mean CBF in cerebrum in white matter and >50% cerebral partial volume-2.0',
    '24376-3.0': 'Mean CBF in cerebrum in white matter and >50% cerebral partial volume-3.0',
    '24361-2.0': 'Mean CBF in cortex in grey matter-2.0',
    '24361-3.0': 'Mean CBF in cortex in grey matter-3.0',
    '24360-2.0': 'Mean CBF in whole brain in grey matter-2.0',
    '24360-3.0': 'Mean CBF in whole brain in grey matter-3.0',
    '24375-2.0': 'Mean CBF in whole brain in white matter-2.0',
    '24375-3.0': 'Mean CBF in whole brain in white matter-3.0',
    }

merged_data.rename(columns = names_to_change, inplace = True)

# Save the cleaned and merged data (biomarkers + ASL-derived information)
merged_data.to_csv(path_ukbiobank + 'merged_biodata_with_cbf.csv', index = False)

#------------------------------------------------------------------------------
# END