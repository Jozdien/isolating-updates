import argparse

import utils

'''
Functions to load data:
    rewards = utils.load_from_json(path + "/rewards.json")
    train_stats = utils.load_mult_json(path + "/log/progress.json")
    weights = utils.load_weights(weights_name, path)
Analysis functions:
    utils.print_stats(rewards)
    utils.rw_plot_exact(rewards, fuel) - fuel bool, determines whether net reward or fuel reward, default False
    utils.rw_plot_fit(rewards, fuel, outliers) - outliers bool, determines whether to include outliers, default True
    utils.train_plot(train_stats, var_name) - var_name str, name of variable to plot, default "rollout/ep_rew_mean"
'''

parser = argparse.ArgumentParser(description='Enter path to folder containing run files.')
parser.add_argument('--path', metavar='path', type=str, help='path to run')
args = parser.parse_args()
path = args.path

rewards = utils.load_from_json(path + "/rewards.json")
train_stats = utils.load_mult_json(path + "/log/progress.json")

utils.print_stats(rewards)
utils.rw_plot_exact(rewards, fuel=False)
utils.rw_plot_exact(rewards, fuel=True)
utils.rw_plot_fit(rewards, fuel=False)
utils.rw_plot_fit(rewards, fuel=True)
utils.train_plot(train_stats, var_name="rollout/ep_rew_mean")