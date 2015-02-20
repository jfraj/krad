"""
Generating the submission file, given the predicted data.

The methods are inspired by the sample_solution.py script 
provided by Alex Kleeman, The Climate Corporation
"""
import csv
import sys
import logging
import argparse
import numpy as np

# configure logging
logger = logging.getLogger("generating_submission")

handler = logging.StreamHandler(sys.stderr)

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)



def generate_submission_file(list_id,submission_data, verbose=True):
    """
    Simple script to generate the submission file.
    verbose will have the progression printed out
    """
    nrow = len(list_id)
    file = open('submission.csv','w')
    # wrap the inputs and outputs in csv interpreters
    writer = csv.writer(file, delimiter=',')

    # the solution header is an Id, then 70 discrete cumulative probabilities
    solution_header = ['Id']
    # Add the fields defining the cumulative probabilites
    solution_header.extend(['Predicted{0}'.format(t) for t in xrange(0, 70)])
    # write the header to file
    writer.writerow(solution_header)

    i = 0
    for id_num, row in zip(list_id,submission_data):
        i += 1
        # write the solution row
        solution_row = [id_num]
        solution_row.extend(row)
        writer.writerow(solution_row)
        # Every 1000 rows send an update to the user for progress tracking.
        if i % 1000 == 0  and verbose:
            logger.info("Completed row %d (%d%%)" %(i, 100*i/nrow))

    return                
