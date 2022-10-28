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

import argparse
import utils

parser = argparse.ArgumentParser(description='Enter path to folder containing run files.')
parser.add_argument('--path', metavar='path', type=str, help='path to run')
args = parser.parse_args()
path = args.path

rewards = utils.load_from_json(path + "/rewards.json")
train_stats = utils.load_mult_json(path + "/log/progress.json")
metadata = utils.load_from_json(path + "/metadata.json")

# utils.print_stats(rewards)
utils.analysis_plots(name=path[path.rfind('/')+1:], rewards=rewards, train_stats=train_stats, metadata=metadata, save=True, show=False)