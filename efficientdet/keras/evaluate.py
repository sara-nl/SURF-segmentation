"""
Evaluation script for CAMELYON17.
"""

import sklearn.metrics
import pandas as ps

import argparse

#----------------------------------------------------------------------------------------------------

def calculate_kappa(reference, submission):
    """
    Calculate inter-annotator agreement with quadratic weighted Kappa.

    Args:
        reference (pandas.DataFrame): List of labels assigned by the organizers.
        submission (pandas.DataFrame): List of labels assigned by participant.

    Returns:
        float: Kappa score.

    Raises:
        ValueError: Unknown stage in reference.
        ValueError: Patient missing from submission.
        ValueError: Unknown stage in submission.
    """

    # The accepted stages are pN0, pN0(i+), pN1mi, pN1, pN2 as described on the website. During parsing all strings converted to lowercase.
    #
    stage_list = ['pn0', 'pn0(i+)', 'pn1mi', 'pn1', 'pn2']

    # Extract the patient pN stages from the tables for evaluation.
    #
    reference_map = {df_row[0]: df_row[1].lower() for _, df_row in reference.iterrows() if df_row[0].lower().endswith('.zip')}
    submission_map = {df_row[0]: df_row[1].lower() for _, df_row in submission.iterrows() if df_row[0].lower().endswith('.zip')}

    # Reorganize data into lists with the same patient order and check consistency.
    #
    reference_stage_list = []
    submission_stage_list = []
    for patient_id, reference_stage in reference_map.items():
        # Check consistency: all stages must be from the official stage list and there must be a submission for each patient in the ground truth.
        #
        submission_stage = submission_map[patient_id].lower()

        if reference_stage not in stage_list:
            raise ValueError('Unknown stage in reference: \'{stage}\''.format(stage=reference_stage))
        if patient_id not in submission_map:
            raise ValueError('Patient missing from submission: \'{patient}\''.format(patient=patient_id))
        if submission_stage not in stage_list:
            raise ValueError('Unknown stage in submission: \'{stage}\''.format(stage=submission_map[patient_id]))

        # Add the pair to the lists.
        #
        reference_stage_list.append(reference_stage)
        submission_stage_list.append(submission_stage)

    # Return the Kappa score.
    #
    return sklearn.metrics.cohen_kappa_score(y1=reference_stage_list, y2=submission_stage_list, labels=stage_list, weights='quadratic')

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str): The parsed reference and submission CSV file paths.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Calculate inter-annotator agreement.')

    argument_parser.add_argument('-r', '--reference',  required=True, type=str, help='reference CSV path')
    argument_parser.add_argument('-s', '--submission', required=True, type=str, help='submission CSV path')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_reference_path = arguments['reference']
    parsed_submission_path = arguments['submission']

    # Print parsed parameters.
    #
    print(argument_parser.description)
    print('Reference: {path}'.format(path=parsed_reference_path))
    print('Submission: {path}'.format(path=parsed_submission_path))

    return parsed_reference_path, parsed_submission_path

#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Parse parameters.
    #
    reference_path, submission_path = collect_arguments()

    # Load tables to Pandas data frames.
    #
    reference_df = ps.read_csv(reference_path)
    submission_df = ps.read_csv(submission_path)

    # Calculate kappa score.
    #
    try:
        kappa_score = calculate_kappa(reference=reference_df, submission=submission_df)
    except Exception as exception:
        print(exception)
    else:
        print('Score: {score}'.format(score=kappa_score))
