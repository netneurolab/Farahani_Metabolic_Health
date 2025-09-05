"""

UK biobank

Combine behavioral data to the already cleaned biomarker + ASL-derived data
The generated csv file is named: path_ukbiobank + 'merged_behavior_with_cbf.csv'.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import pandas as pd
from globals import path_ukbiobank

#------------------------------------------------------------------------------
# Load blood test + demographics data, clean them, and add BMI as one of the measures here
#------------------------------------------------------------------------------

main_data = pd.read_csv(path_ukbiobank + 'data_082025_withimaging.csv') # raw data

# Define interesting labels - behaviors
interesting_labels = [
    'eid',
    'age_completed_full_time_education_845-2.0',
    'number_of_incorrect_matches_in_round_399-2.1',
    'number_of_incorrect_matches_in_round_399-2.2',
    'number_of_incorrect_matches_in_round_399-2.3',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.0',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.1',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.2',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.3',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.4',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.5',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.6',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.7',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.8',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.9',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.10',
    'duration_to_first_press_of_snap-button_in_each_round_404-2.11',
    'duration_of_walks_874-2.0',
    'sleep_duration_1160-0.0',
    'sleep_duration_1160-2.0',
    'sleeplessness__insomnia_1200-0.0',
    'sleeplessness__insomnia_1200-2.0',
    'snoring_1210-0.0',
    'snoring_1210-2.0',
    'current_tobacco_smoking_1239-2.0',
    'past_tobacco_smoking_1249-2.0',
    'average_weekly_beer_plus_cider_intake_1588-2.0',
    'weight_change_compared_with_1_year_ago_2306-2.0',
    'maximum_digits_remembered_correctly_4282-2.0',
    'number_of_depression_episodes_4620-2.0',
    'attempted_fluid_intelligence_fi_test._4924-2.0',
    'duration_spent_answering_each_puzzle_6333-2.0',
    'duration_spent_answering_each_puzzle_6333-2.1',
    'duration_spent_answering_each_puzzle_6333-2.2',
    'duration_spent_answering_each_puzzle_6333-2.3',
    'duration_spent_answering_each_puzzle_6333-2.4',
    'duration_spent_answering_each_puzzle_6333-2.5',
    'duration_spent_answering_each_puzzle_6333-2.6',
    'duration_spent_answering_each_puzzle_6333-2.7',
    'duration_spent_answering_each_puzzle_6333-2.8',
    'duration_spent_answering_each_puzzle_6333-2.9',
    'duration_spent_answering_each_puzzle_6333-2.10',
    'duration_spent_answering_each_puzzle_6333-2.11',
    'duration_spent_answering_each_puzzle_6333-2.12',
    'duration_spent_answering_each_puzzle_6333-2.13',
    'duration_spent_answering_each_puzzle_6333-2.14',
    'prospective_memory_result_20018-2.0',
    'smoking_status_20116-0.0',
    'smoking_status_20116-2.0',
    'time_spent_using_computer_1080-2.0',
    'time_spent_using_computer_1080-2.0',
    'weight_change_compared_with_1_year_ago_2306-2.0',
    'number_of_daysweek_of_moderate_physical_activity_10+_minutes_884-2.0',
    'duration_of_moderate_activity_894-2.0',
    'number_of_daysweek_of_vigorous_physical_activity_10+_minutes_904-0.0',
    'number_of_daysweek_of_vigorous_physical_activity_10+_minutes_904-2.0',
    'number_of_puzzles_correctly_solved_6373-2.0',
    'fluid_intelligence_score_20016-2.0',
    'prospective_memory_result_20018-2.0',
    'frequency_of_drinking_alcohol_20414-0.0',
    'mean_time_to_correctly_identify_matches_20023-0.0',
    'mean_time_to_correctly_identify_matches_20023-2.0'
]
# Only keep the interesting columns
selected_data = main_data[interesting_labels]

# Read the already cleaned ASL data
merged_data = pd.read_csv(path_ukbiobank + 'merged_biodata_with_cbf.csv') # This is created by the code in the previous step

# Merge the selected behavioral data onto merged_data (keeping merged_data subjects)
merged_with_behavior = pd.merge(merged_data, selected_data, on = 'eid', how = 'left')

# Save the dataframe
merged_with_behavior.to_csv(path_ukbiobank + 'merged_behavior_with_cbf.csv',
                            index = False)

#------------------------------------------------------------------------------
# END