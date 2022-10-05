import json
import argparse
import statistics

parser = argparse.ArgumentParser(description='Enter path to folder containing run files.')
parser.add_argument('--path', metavar='path', type=str, help='path to run')
args = parser.parse_args()
path = args.path

def load_rewards(path=path):
    if path[-5:] != '.json':
        path += '/rewards.json'
    return json.load(open(path))

def load_weights(weights, path=path):
    return json.load(open(path + '/' + weights))



pre_update_rews = rewards['pre_update']
post_update_rews = rewards['post_update']
sub_update_rews = rewards['sub_update']

pre_update_fuel_rews = pre_update_rews['fuel_reward']
post_update_fuel_rews = post_update_rews['fuel_reward']
sub_update_fuel_rews = sub_update_rews['fuel_reward']

print('Pre-update fuel rewards mean: ', statistics.mean(pre_update_fuel_rews))
print('Post-update fuel rewards mean: ', statistics.mean(post_update_fuel_rews))
print('Sub-update fuel rewards mean: ', statistics.mean(sub_update_fuel_rews))

print('Pre-update fuel rewards std: ', statistics.stdev(pre_update_fuel_rews))
print('Post-update fuel rewards std: ', statistics.stdev(post_update_fuel_rews))
print('Sub-update fuel rewards std: ', statistics.stdev(sub_update_fuel_rews))

print(pre_update_fuel_rews.count(0))
print(post_update_fuel_rews.count(0))
print(sub_update_fuel_rews.count(0))