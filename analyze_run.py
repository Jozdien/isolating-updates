import json
import argparse
import statistics

parser = argparse.ArgumentParser(description='Enter path to folder containing run files.')
parser.add_argument('--path', metavar='path', type=str, help='path to run')
args = parser.parse_args()
path = args.path

def load_rewards(path=path):
    '''
    Returns a dictionary containing the rewards from a single run.
    Keys: 'pre_update', 'post_update', and 'sub_update'.
    Each subdict contains net reward from each step, and fuel reward from each step.
    '''
    if path[-5:] != '.json':
        path += '/rewards.json'
    return json.load(open(path))

def load_weights(weights, path=path):
    '''
    Returns a dictionary containing the weights from a single run.
    '''
    return json.load(open(path + '/' + weights))

def phase_rewards(rewards):
    '''
    Returns three dicts for three phases, containing net rewards from each step and fuel rewards from each step.
    '''
    return rewards['pre_update'], rewards['post_update'], rewards['sub_update']

def phase_fuel_rewards(rewards):
    '''
    Returns three lists for three phases, containing fuel rewards from each step.
    '''
    return rewards['pre_update']['fuel_reward'], rewards['post_update']['fuel_reward'], rewards['sub_update']['fuel_reward']

def fuel_means(rewards):
    '''
    Returns mean fuel reward for each phase.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.mean(pre), statistics.mean(post), statistics.mean(sub)

def fuel_stds(rewards):
    '''
    Returns standard deviation of fuel reward for each phase.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.stdev(pre), statistics.stdev(post), statistics.stdev(sub)

def fuel_zeros(rewards):
    pre, post, sub = phase_fuel_rewards(rewards)
    return pre.count(0), post.count(0), sub.count(0)

def no_zeros_mean(lst):
    return statistics.mean([x for x in lst if x != 0])

def no_zeros_std(lst):
    return statistics.stdev([x for x in lst if x != 0])

def fuel_means_no_zeros(rewards):
    pre, post, sub = phase_fuel_rewards(rewards)
    return no_zeros_mean(pre), no_zeros_mean(post), no_zeros_mean(sub)

def fuel_stds_no_zeros(rewards):
    pre, post, sub = phase_fuel_rewards(rewards)
    return no_zeros_std(pre), no_zeros_std(post), no_zeros_std(sub)

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