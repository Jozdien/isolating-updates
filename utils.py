import torch
import numpy as np
import os
import datetime
import copy
import json
import statistics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def test_model(env, model, rewards_dict, phase, render=False, num_episodes=1000):
    '''
    Runs a model on the given environment for num_episodes.
    Stores in rewards_dict the total reward as well as the fuel reward for each episode, under the key phase.
    '''
    obs = env.reset()
    for i in range(num_episodes):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards_dict[phase]['total_reward'].append(reward)
        rewards_dict[phase]['fuel_reward'].append(info['fuel_reward'])
        if render:
            env.render()
        if done:
            obs = env.reset()

def mkdir_timestamped(path="runs/"):
    '''
    Creates a directory with the current date and time.
    '''
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
    name = os.path.join(os.getcwd(), path, time)
    os.mkdir(name)
    return path + "/" + time

def save_metadata(metadata, path="runs/"):
    '''
    Saves the metadata of a run to a json file.
    '''
    with open(path + "metadata.json", 'w') as f:
        json.dump(metadata, f)

def save_weights(weights, path="runs/", name="weights.json"):
    '''
    Saves the weights of a model to a json file.
    '''
    with open(path + name, 'w') as f:
        json.dump(weights, f)

def saveable(model):
    '''
    Returns a model that can be saved to a json file.
    '''
    return {key: value.tolist() for key, value in model.items()}

def save_to_json(obj, path="runs/", name="obj.json"):
    '''
    Saves an object to a json file.
    '''
    with open(path + name, 'w') as f:
        json.dump(obj, f)

def to_dict(key, obj):
    '''
    Returns a dict of the key-obj pair.
    '''
    return {key: obj}

def true_copy(obj):
    '''
    Returns a true copy of an object.
    '''
    return copy.deepcopy(obj)

def weights_eq(model1, model2):
    '''
    Returns True if the weights of two models are equal.
    '''
    if model1.keys() != model2.keys():
        return False
    for key in model1.keys():
        if not torch.equal(model1[key], model2[key]):
            return False
    return True

def weight_diff(model1, model2):
    '''
    Returns the difference in weights between two models of identical architecture.
    '''
    return {key: torch.sub(model1[key], model2[key]) for key in model1.keys()}

def get_weights(model):
    '''
    Returns the weights of the model as an odict with the following keys:
    'mlp_extractor.policy_net.0.weight'
    'mlp_extractor.policy_net.0.bias'
    'mlp_extractor.policy_net.2.weight'
    'mlp_extractor.policy_net.2.bias'
    'mlp_extractor.value_net.0.weight'
    'mlp_extractor.value_net.0.bias'
    'mlp_extractor.value_net.2.weight'
    'mlp_extractor.value_net.2.bias'
    'action_net.weight'
    'action_net.bias'
    'value_net.weight'
    'value_net.bias'
    '''
    return model.policy.state_dict()

def heatmap_sep(model, title="Plot", set_abs=True, show=True, save=False, save_as_prefix="plot", path="runs/"):
    '''
    Plots separate heatmaps of every weight tensor in a model.
    '''
    if save_as_prefix.endswith(".png"):
        save_as_prefix = save_as_prefix[:-4]
    if not os.path.exists(path):
        os.mkdir(path)
    for count, (name, weights) in enumerate(model.items()):
        if len(weights.shape) == 1:
            weights = weights.reshape(1, -1)
        fig = plt.figure()
        ax = plt.subplot()
        ax.set_title(name)
        if set_abs:  # SET WEIGHTS TO ABS FOR PLOTTING
            weights = np.absolute(weights)
        ax.imshow(weights, cmap='hot', interpolation='nearest')
        if save:
            plt.savefig(path + save_as_prefix + "_" + name + ".png", bbox_inches='tight')
    if show:
        plt.show()

def load_rewards(path):
    '''
    Returns a dictionary containing the rewards from a single run.
    Keys: 'pre_update', 'post_update', and 'sub_update'.
    Each subdict contains net reward from each step, and fuel reward from each step.
    '''
    if path[-5:] != '.json':
        path += '/rewards.json'
    return json.load(open(path))

def load_weights(weights, path):
    '''
    Returns a dictionary containing the weights from a single run.
    '''
    return json.load(open(path + '/' + weights))

def phase_rewards(rewards):
    '''
    Returns three dicts for three phases, containing net rewards from each step and fuel rewards from each step.
    '''
    return rewards['pre_update'], rewards['post_update'], rewards['sub_update']

def phase_total_rewards(rewards):
    '''
    Returns three lists for three phases, containing the total reward from each step.
    '''
    return rewards['pre_update']['total_reward'], rewards['post_update']['total_reward'], rewards['sub_update']['total_reward']

def phase_fuel_rewards(rewards):
    '''
    Returns three lists for three phases, containing fuel rewards from each step.
    '''
    return rewards['pre_update']['fuel_reward'], rewards['post_update']['fuel_reward'], rewards['sub_update']['fuel_reward']

def reward_means(rewards):
    '''
    Returns the mean reward for each phase.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return statistics.mean(pre), statistics.mean(post), statistics.mean(sub)

def fuel_means(rewards):
    '''
    Returns mean fuel reward for each phase.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.mean(pre), statistics.mean(post), statistics.mean(sub)

def reward_stds(rewards):
    '''
    Returns the standard deviation of the reward for each phase.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return statistics.stdev(pre), statistics.stdev(post), statistics.stdev(sub)

def fuel_stds(rewards):
    '''
    Returns standard deviation of fuel reward for each phase.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.stdev(pre), statistics.stdev(post), statistics.stdev(sub)

def reward_variance(rewards):
    '''
    Returns the variance of the reward for each phase.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return statistics.variance(pre), statistics.variance(post), statistics.variance(sub)

def fuel_variance(rewards):
    '''
    Returns variance of fuel reward for each phase.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.variance(pre), statistics.variance(post), statistics.variance(sub)

def reward_zeros_count(rewards):
    '''
    Returns the number of zero rewards for each phase.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return pre.count(0), post.count(0), sub.count(0)

def reward_zeros_percent(rewards):
    '''
    Returns the percentage of zero rewards for each phase.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return pre.count(0) / len(pre), post.count(0) / len(post), sub.count(0) / len(sub)

def fuel_zeros_count(rewards):
    '''
    Returns number of 0 fuel rewards in each phase.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return pre.count(0), post.count(0), sub.count(0)

def fuel_zeros_percent(rewards):
    '''
    Returns percentage of 0 fuel rewards in each phase.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return pre.count(0) / len(pre), post.count(0) / len(post), sub.count(0) / len(sub)

def remove_zeros(lst):
    '''
    Returns a list with all 0s removed.
    '''
    return [x for x in lst if x != 0]

def reward_means_no_zeros(rewards):
    '''
    Returns mean reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return statistics.mean(remove_zeros(pre)), statistics.mean(remove_zeros(post)), statistics.mean(remove_zeros(sub))

def fuel_means_no_zeros(rewards):
    '''
    Returns mean fuel reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.mean(remove_zeros(pre)), statistics.mean(remove_zeros(post)), statistics.mean(remove_zeros(sub))

def reward_stds_no_zeros(rewards):
    '''
    Returns standard deviation of reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return statistics.stdev(remove_zeros(pre)), statistics.stdev(remove_zeros(post)), statistics.stdev(remove_zeros(sub))

def fuel_stds_no_zeros(rewards):
    '''
    Returns standard deviation of fuel reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.stdev(remove_zeros(pre)), statistics.stdev(remove_zeros(post)), statistics.stdev(remove_zeros(sub))

def reward_variance_no_zeros(rewards):
    '''
    Returns variance of reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_total_rewards(rewards)
    return statistics.variance(remove_zeros(pre)), statistics.variance(remove_zeros(post)), statistics.variance(remove_zeros(sub))

def fuel_variance_no_zeros(rewards):
    '''
    Returns variance of fuel reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.variance(remove_zeros(pre)), statistics.variance(remove_zeros(post)), statistics.variance(remove_zeros(sub))