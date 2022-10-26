import torch
import numpy as np
import os
import datetime
import copy
import json
import statistics
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
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


def load_from_json(path):
    '''
    Returns an object from a json file.
    '''
    return json.load(open(path))

def load_mult_json(path):
    '''
    Loads from a file containing multiple JSON objects.
    '''
    with open(path, encoding="utf-8") as f:
        txt = f.read().lstrip()
    decoder = json.JSONDecoder()
    result = []
    while txt:
        data, pos = decoder.raw_decode(txt)
        result.append(data)
        txt = txt[pos:].lstrip()
    return result

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

def fuel_means_no_zeros(rewards):
    '''
    Returns mean fuel reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.mean(remove_zeros(pre)), statistics.mean(remove_zeros(post)), statistics.mean(remove_zeros(sub))

def fuel_stds_no_zeros(rewards):
    '''
    Returns standard deviation of fuel reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.stdev(remove_zeros(pre)), statistics.stdev(remove_zeros(post)), statistics.stdev(remove_zeros(sub))

def fuel_variance_no_zeros(rewards):
    '''
    Returns variance of fuel reward for each phase, with 0s removed.
    '''
    pre, post, sub = phase_fuel_rewards(rewards)
    return statistics.variance(remove_zeros(pre)), statistics.variance(remove_zeros(post)), statistics.variance(remove_zeros(sub))

def remove_outliers(data, m=100):
    '''
    Removes outliers from a list.
    Increase m to include more outliers.
    '''
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def print_stats(rewards):
    '''
    Prints a select number of stats about the net rewards and fuel rewards from each phase.
    '''
    rw_means = reward_means(rewards)
    rw_stds = reward_stds(rewards)
    rw_vars = reward_variance(rewards)

    fl_zeros_percent = fuel_zeros_percent(rewards)
    fl_means = fuel_means(rewards)
    fl_means_no_zeros = fuel_means_no_zeros(rewards)
    fl_stds = fuel_stds(rewards)
    fl_vars = fuel_variance(rewards)
    fl_vars_no_zeros = fuel_variance_no_zeros(rewards)

    print('Reward means: {}'.format(rw_means))
    print('Reward standard deviations: {}'.format(rw_stds))
    print('Reward variances: {}'.format(rw_vars))

    print('Fuel zeros percent: {}'.format(fl_zeros_percent))
    print('Fuel means: {}'.format(fl_means))
    print('Fuel means (no zeros): {}'.format(fl_means_no_zeros))
    print('Fuel standard deviations: {}'.format(fl_stds))
    print('Fuel variances: {}'.format(fl_vars))
    print('Fuel variances (no zeros): {}'.format(fl_vars_no_zeros))

def stats_dict(rewards):
    '''
    Returns the stats from the above function in dict form.
    '''
    return {
        'Reward means': reward_means(rewards), 
        'Reward standard deviations': reward_stds(rewards),
        'Reward variances': reward_variance(rewards),
        'Fuel zeros percent': fuel_zeros_percent(rewards),
        'Fuel means': fuel_means(rewards),
        'Fuel means (no zeros)': fuel_means_no_zeros(rewards),
        'Fuel standard deviations': fuel_stds(rewards),
        'Fuel variances': fuel_variance(rewards),
        'Fuel variances (no zeros)': fuel_variance_no_zeros(rewards)
    }

def rw_plot_exact(rewards, fuel=False, show=True):
    '''
    Plots the total or fuel reward for each phase.
    '''
    if fuel:
        pre, post, sub = phase_fuel_rewards(rewards)
    else:
        pre, post, sub = phase_total_rewards(rewards)
    plt.plot(pre, label='pre')
    plt.plot(post, label='post')
    plt.plot(sub, label='sub')
    plt.xlabel('Timesteps')
    if fuel:
        plt.ylabel('Fuel reward')
        plt.title('Fuel reward on testing after each phase')
    else:
        plt.ylabel('Total reward')
        plt.title('Total reward on testing after each phase')
    plt.legend()
    if show:
        plt.show()

def rw_plot_fit(rewards, fuel=False, outliers=True, show=True):
    '''
    Plots the total or fuel reward for each phase, with a fitted curve line.
    '''
    if fuel:
        pre, post, sub = phase_fuel_rewards(rewards)
    else:
        pre, post, sub = phase_total_rewards(rewards)
        if not outliers:
            pre, post, sub = remove_outliers(pre), remove_outliers(post), remove_outliers(sub)
    y = pre
    x = [i for i in range(len(y))]
    x_y_spline = make_interp_spline(x, y)
    if fuel:
        x_new = np.linspace(x[0], x[-1], 40)
    else:
        x_new = np.linspace(x[0], x[-1], 500)
    y_new = x_y_spline(x_new)
    plt.plot(x_new, y_new, label='pre')
    y = post
    x = [i for i in range(len(y))]
    x_y_spline = make_interp_spline(x, y)
    if fuel:
        x_new = np.linspace(x[0], x[-1], 40)
    else:
        x_new = np.linspace(x[0], x[-1], 100)
    y_new = x_y_spline(x_new)
    plt.plot(x_new, y_new, label='post')
    y = sub
    x = [i for i in range(len(y))]
    x_y_spline = make_interp_spline(x, y)
    if fuel:
        x_new = np.linspace(x[0], x[-1], 40)
    else:
        x_new = np.linspace(x[0], x[-1], 100)
    y_new = x_y_spline(x_new)
    plt.plot(x_new, y_new, label='sub')
    plt.xlabel('Timesteps')
    if fuel:
        plt.ylabel('Fuel reward')
        plt.title('Fitted curve of fuel reward on testing after each phase')
    else:
        plt.ylabel('Total reward')
        if not outliers:
            plt.title('Fitted curve of total reward on testing after each phase (no outliers)')
        else:
            plt.title('Fitted curve of total reward on testing after each phase')
    plt.legend()
    if show:
        plt.show()

def train_plot(train_stats, var_name='rollout/ep_rew_mean', show=True):
    var = [episode[var_name] for episode in train_stats]
    tsteps = [episode['time/total_timesteps'] for episode in train_stats]
    crossover = [i for i, n in enumerate(tsteps) if n == 2048][1]
    tsteps[crossover:] = [n + tsteps[crossover - 1] for n in tsteps[crossover:]]
    plt.plot(tsteps[:crossover], var[:crossover], label='First phase')
    plt.plot(tsteps[crossover:], var[crossover:], 'b-', label='Second phase')
    plt.plot([tsteps[crossover - 1], tsteps[crossover]], [var[crossover - 1], var[crossover]], 'b-')
    plt.axvline(x=tsteps[crossover-1], color='r', linestyle='-', linewidth=0.5)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Reward curve across both phases')
    plt.legend()
    if show:
        plt.show()

def analysis_plots(name, rewards, train_stats, metadata, show=True, save=False, path="analysis/"):
    '''
    Composite analysis plots.
    '''
    fig, ax = plt.subplots(6, 1, figsize=(25, 60))  # Just to set the figure size
    plt.subplots_adjust(top=0.88, bottom=0.05)

    plt_title = 'Run: {}'.format(name)
    plt.text(0, 7.6, plt_title, fontsize=30, transform=plt.gca().transAxes)
    ts_info = "First training: {} timesteps\nSecond training: {} timesteps".format(metadata['FIRST_TRAIN_TIMESTEPS'], metadata['SECOND_TRAIN_TIMESTEPS'])
    plt.text(0.75, 7.51, ts_info, fontsize=20, transform=plt.gca().transAxes, bbox=dict(facecolor='none', edgecolor='black', linewidth=3, pad=40.0))
    stats = stats_dict(rewards)
    stats_str = ""
    for (k, v) in stats.items():
        stats_str += k + ": " + str(v)[1:-1] + "\n"
    plt.text(0, 7.15, stats_str, fontsize=16, transform=plt.gca().transAxes)

    plt.subplot(6, 1, 1)
    rw_plot_fit(rewards, show=False)
    plt.subplot(6, 1, 2)
    rw_plot_fit(rewards, outliers=False, show=False)
    plt.subplot(6, 1, 3)
    rw_plot_exact(rewards, show=False)
    plt.subplot(6, 1, 4)
    rw_plot_fit(rewards, fuel=True, show=False)
    plt.subplot(6, 1, 5)
    rw_plot_exact(rewards, fuel=True, show=False)
    plt.subplot(6, 1, 6)
    train_plot(train_stats, show=False)

    if save:
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path + name + '_analysis.png')
    if show:
        plt.show()