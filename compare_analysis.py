'''
Code to visualize and compare composite plots from multiple runs with linked pan and zoom.
'''

import os
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils

parser = argparse.ArgumentParser(description='Enter runs to compare.')
parser.add_argument('--runs', metavar='runs', type=str, nargs='+', help='runs to compare')
args = parser.parse_args()
runs = args.runs
if runs == None:
    runs = [
        'analysis/2022-10-20_19-11-51_analysis.png', 
        'analysis/2022-10-20_19-34-20_analysis.png',
        'analysis/2022-10-20_20-35-14_analysis.png',
        'analysis/2022-10-20_21-09-58_analysis.png',
        'analysis/2022-10-21_00-40-54_analysis.png',
        'analysis/2022-10-21_01-17-08_analysis.png',
        'analysis/2022-10-21_01-26-50_analysis.png',
        'analysis/2022-10-21_01-37-13_analysis.png',
    ]
elif runs[0] == 'all':
    runs = ["analysis/" + run for run in os.listdir('analysis')]

utils.compare_plots(runs)