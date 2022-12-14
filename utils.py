import torch
import numpy as np
import os
import datetime
import copy
import json
import math
import statistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import make_interp_spline
from matplotlib.gridspec import GridSpec


def test_model(env, model, rewards_dict, phase, render=False, num_episodes=5000):
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

def update_rundir(name, metadata, path="run_directory.json"):
    '''
    Adds a run to the run directory according to its metadata.
    '''
    wrappers = "WRP_1:{} WRP_2:{} TEST_ENV:{} WRP_3:{}".format(metadata['FIRST_WRAPPER'], metadata['SECOND_WRAPPER'], metadata['TEST_ENV'], metadata['SALVAGE_WRAPPER'])
    timesteps = "TSTEPS_1:{} TSTEPS_2:{} TSTEPS_3:{}".format(metadata['FIRST_TRAIN_TIMESTEPS'], metadata['SECOND_TRAIN_TIMESTEPS'], metadata['SALVAGE_TIMESTEPS'])
    with open(path, 'r') as f:
        try:
            curr = json.load(f)
        except:
            curr = {}
        if wrappers in curr and timesteps in curr[wrappers]:
            curr[wrappers][timesteps].append(name)
        else:
            if wrappers not in curr:
                curr[wrappers] = {}
            curr[wrappers][timesteps] = [name]
    with open(path, 'w') as f:
        json.dump(curr, f, indent=4)

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
    return rewards['init'], rewards['pre_update'], rewards['post_update'], rewards['sub_update'], rewards['salvage']

def phase_total_rewards(rewards):
    '''
    Returns three lists for three phases, containing the total reward from each step.
    '''
    return rewards['init']['total_reward'], rewards['pre_update']['total_reward'], rewards['post_update']['total_reward'], rewards['sub_update']['total_reward'], rewards['salvage']['total_reward']

def phase_fuel_rewards(rewards):
    '''
    Returns three lists for three phases, containing fuel rewards from each step.
    '''
    return rewards['init']['fuel_reward'], rewards['pre_update']['fuel_reward'], rewards['post_update']['fuel_reward'], rewards['sub_update']['fuel_reward'], rewards['salvage']['fuel_reward']

def reward_means(rewards):
    '''
    Returns the mean reward for each phase.
    '''
    init, pre, post, sub, salvage = phase_total_rewards(rewards)
    return statistics.mean(init), statistics.mean(pre), statistics.mean(post), statistics.mean(sub), statistics.mean(salvage)

def fuel_means(rewards):
    '''
    Returns mean fuel reward for each phase.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return statistics.mean(init), statistics.mean(pre), statistics.mean(post), statistics.mean(sub), statistics.mean(salvage)

def reward_stds(rewards):
    '''
    Returns the standard deviation of the reward for each phase.
    '''
    init, pre, post, sub, salvage = phase_total_rewards(rewards)
    return statistics.stdev(init), statistics.stdev(pre), statistics.stdev(post), statistics.stdev(sub), statistics.stdev(salvage)

def fuel_stds(rewards):
    '''
    Returns standard deviation of fuel reward for each phase.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return statistics.stdev(init), statistics.stdev(pre), statistics.stdev(post), statistics.stdev(sub), statistics.stdev(salvage)

def reward_variance(rewards):
    '''
    Returns the variance of the reward for each phase.
    '''
    init, pre, post, sub, salvage = phase_total_rewards(rewards)
    return statistics.variance(init), statistics.variance(pre), statistics.variance(post), statistics.variance(sub), statistics.variance(salvage)

def fuel_variance(rewards):
    '''
    Returns variance of fuel reward for each phase.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return statistics.variance(init), statistics.variance(pre), statistics.variance(post), statistics.variance(sub), statistics.variance(salvage)

def fuel_zeros_count(rewards):
    '''
    Returns number of 0 fuel rewards in each phase.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return init.count(0), pre.count(0), post.count(0), sub.count(0), salvage.count(0)

def fuel_zeros_percent(rewards):
    '''
    Returns percentage of 0 fuel rewards in each phase.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return init.count(0)/len(init), pre.count(0)/len(pre), post.count(0)/len(post), sub.count(0)/len(sub), salvage.count(0)/len(salvage)

def remove_zeros(lst):
    '''
    Returns a list with all 0s removed.
    '''
    return [x for x in lst if x != 0]

def fuel_means_no_zeros(rewards):
    '''
    Returns mean fuel reward for each phase, with 0s removed.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return statistics.mean(remove_zeros(init)), statistics.mean(remove_zeros(pre)), statistics.mean(remove_zeros(post)), statistics.mean(remove_zeros(sub)), statistics.mean(remove_zeros(salvage))

def fuel_stds_no_zeros(rewards):
    '''
    Returns standard deviation of fuel reward for each phase, with 0s removed.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return statistics.stdev(remove_zeros(init)), statistics.stdev(remove_zeros(pre)), statistics.stdev(remove_zeros(post)), statistics.stdev(remove_zeros(sub)), statistics.stdev(remove_zeros(salvage))

def fuel_variance_no_zeros(rewards):
    '''
    Returns variance of fuel reward for each phase, with 0s removed.
    '''
    init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    return statistics.variance(remove_zeros(init)), statistics.variance(remove_zeros(pre)), statistics.variance(remove_zeros(post)), statistics.variance(remove_zeros(sub)), statistics.variance(remove_zeros(salvage))

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
        init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    else:
        init, pre, post, sub, salvage = phase_total_rewards(rewards)
    plt.plot(init, label='initial')
    plt.plot(pre, label='pre')
    plt.plot(post, label='post')
    plt.plot(sub, label='sub')
    plt.plot(salvage, label='salvage')
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
        init, pre, post, sub, salvage = phase_fuel_rewards(rewards)
    else:
        init, pre, post, sub, salvage = phase_total_rewards(rewards)
        if not outliers:
            init, pre, post, sub, salvage = remove_outliers(init), remove_outliers(pre), remove_outliers(post), remove_outliers(sub), remove_outliers(salvage)
    y = init
    x = [i for i in range(len(y))]
    x_y_spline = make_interp_spline(x, y)
    if fuel:
        x_new = np.linspace(x[0], x[-1], 40)
    else:
        x_new = np.linspace(x[0], x[-1], 500)
    y_new = x_y_spline(x_new)
    plt.plot(x_new, y_new, label='initial')
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
    y = salvage
    x = [i for i in range(len(y))]
    x_y_spline = make_interp_spline(x, y)
    if fuel:
        x_new = np.linspace(x[0], x[-1], 40)
    else:
        x_new = np.linspace(x[0], x[-1], 100)
    y_new = x_y_spline(x_new)
    plt.plot(x_new, y_new, label='salvage')
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
    plt.title('Training reward curve across both phases')
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

def query_runs(FIRST_WRAPPER, SECOND_WRAPPER, TEST_ENV, SALVAGE_WRAPPER, 
                FIRST_TRAIN_TIMESTEPS, SECOND_TRAIN_TIMESTEPS, SALVAGE_TIMESTEPS, 
                ALL_TS=False, rundir='run_directory.json'):
    '''
    Returns names of all runs matching the query.
    '''
    wrappers_key = 'WRP_1:{} WRP_2:{} TEST_ENV:{} WRP_3:{}'.format(FIRST_WRAPPER, SECOND_WRAPPER, TEST_ENV, SALVAGE_WRAPPER)
    if not ALL_TS:
        tsteps_key = 'TSTEPS_1:{} TSTEPS_2:{} TSTEPS_3:{}'.format(FIRST_TRAIN_TIMESTEPS, SECOND_TRAIN_TIMESTEPS, SALVAGE_TIMESTEPS)
    with open(rundir, 'r') as f:
        if ALL_TS:
            return sum(json.load(f)[wrappers_key].values(), [])
        return json.load(f)[wrappers_key][tsteps_key]

def compare_stats(runs, exclude=[]):
    '''
    Creates a comparison plot of stats from multiple runs.
    '''
    stats = []
    for run in runs:
        stats.append(stats_dict(load_from_json(run + '/rewards.json')))
    for key in stats[0].keys():
        if key in exclude:
            continue
        fig = plt.figure()
        plt.title(key)
        x_axis = ['init', 'pre', 'post', 'sub', 'salvage']
        vals = []
        for (i, stat) in enumerate(stats):
            if key == 'Reward means':  # Just removing the init phase here for better visualization
                x_axis = ['pre', 'post', 'sub', 'salvage']
                stat[key] = stat[key][1:]
            plt.plot(x_axis, stat[key], linewidth=0.5, label=runs[i])
            vals.append(list(stat[key]))
        avg = np.mean(vals, axis=0)
        plt.plot(avg, linestyle='--', linewidth=1.5, color='black', label='Mean')
        plt.legend(prop={'size': 5})
    plt.show()

def compare_plots(runs):
    '''
    Puts all the analysis plots of the runs in a single plot, with shared panning and zooming for easier analysis.
    Arg runs: path to analysis img.
    '''
    imgs = [mpimg.imread(run) for run in runs]
    num_imgs = len(imgs)
    ipr = 2  # Images per row
    rows = math.ceil(num_imgs / ipr) if num_imgs >= ipr else 1
    cols = ipr if num_imgs > ipr else num_imgs

    fig, axes = plt.subplots(rows, cols, figsize=[25, 60], sharex=True, sharey=True)
    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=-0.05, wspace=-0.25)
    fig.tight_layout()

    for (i, img) in enumerate(imgs):
        if rows == 1:
            axes[i].imshow(img, aspect='auto')
        else:
            axes[int(i / ipr)][i % ipr].imshow(img, aspect='auto')
    [ax.set_axis_off() for ax in axes.ravel()]

    plt.show()