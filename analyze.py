'''
Code for analyzing the results of a run.
Currently combines all results into a composite plot which is saved.
Below are the other functions that could plausibly be useful for analysis.

Functions to load data:
    rewards = utils.load_from_json(path + "/rewards.json")
    train_stats = utils.load_mult_json(path + "/log/progress.json")
    weights = utils.load_weights(weights_name, path)
    metadata = utils.load_from_json(path + "/metadata.json")
Analysis functions:
    utils.print_stats(rewards)
    utils.rw_plot_exact(rewards, fuel) - fuel bool, determines whether net reward or fuel reward, default False
    utils.rw_plot_fit(rewards, fuel, outliers) - outliers bool, determines whether to include outliers, default True
    utils.train_plot(train_stats, var_name) - var_name str, name of variable to plot, default "rollout/ep_rew_mean"
'''

import os
import argparse
import utils

parser = argparse.ArgumentParser(description='Enter paths to runs for analysis')
parser.add_argument('--multiple', action=argparse.BooleanOptionalAction)  # default False
parser.add_argument('--runs', metavar='runs', type=str, nargs='+', help='paths to run folders')
args = parser.parse_args()
multiple = args.multiple
runs = args.runs

if multiple:
    if runs == None:
        runs = [
            'runs/2022-10-20_19-11-51',
            'runs/2022-10-20_19-34-20',
            'runs/2022-10-20_20-35-14',
            'runs/2022-10-20_21-09-58',
            'runs/2022-10-21_00-40-54',
            'runs/2022-10-21_01-17-08',
            'runs/2022-10-21_01-26-50',
            'runs/2022-10-21_01-37-13',
        ]
    elif runs[0] == 'all':
        runs = ["runs/" + run for run in os.listdir('runs')]
    analyses = ['analysis/' + run.split('/')[1] + '_analysis.png' for run in runs]

    utils.compare_stats(runs, exclude=['Reward standard deviations', 'Fuel standard deviations', ])
    # utils.compare_plots(analyses)
else:
    if runs == None:
        runs = ['runs/' + os.listdir('runs')[0]]

    path = runs[0]
    rewards = utils.load_from_json(path + "/rewards.json")
    train_stats = utils.load_mult_json(path + "/log/progress.json")
    metadata = utils.load_from_json(path + "/metadata.json")

    # utils.print_stats(rewards)
    utils.analysis_plots(name=path[path.rfind('/')+1:], rewards=rewards, train_stats=train_stats, metadata=metadata, save=True, show=False)