import argparse

import utils

parser = argparse.ArgumentParser(description='Enter path to folder containing run files.')
parser.add_argument('--path', metavar='path', type=str, help='path to run')
args = parser.parse_args()
path = args.path

rewards = utils.load_rewards(path)

reward_means = utils.reward_means(rewards)
reward_stds = utils.reward_stds(rewards)
reward_variance = utils.reward_variance(rewards)

fuel_zeros_percent = utils.fuel_zeros_percent(rewards)
fuel_means = utils.fuel_means(rewards)
fuel_stds = utils.fuel_stds(rewards)
fuel_variance = utils.fuel_variance(rewards)
fuel_variance_no_zeros = utils.fuel_variance_no_zeros(rewards)

print('Reward zeros percent: {}'.format(reward_zeros_percent))
print('Reward means: {}'.format(reward_means))
print('Reward standard deviations: {}'.format(reward_stds))
print('Reward variances: {}'.format(reward_variance))
print('Reward variances (no zeros): {}'.format(reward_variance_no_zeros))
print('Fuel zeros percent: {}'.format(fuel_zeros_percent))
print('Fuel means: {}'.format(fuel_means))
print('Fuel standard deviations: {}'.format(fuel_stds))
print('Fuel variances: {}'.format(fuel_variance))
print('Fuel variances (no zeros): {}'.format(fuel_variance_no_zeros))