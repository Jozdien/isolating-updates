import torch
import numpy as np
import os
import datetime
import copy
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def test_model(env, model, rewards_dict, phase, render=False, num_episodes=1000):
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
            plt.savefig(path + save_as_prefix + name + ".png", bbox_inches='tight')
    if show:
        plt.show()

def heatmap(model, title="Plot", set_abs=True, save=False, save_as="plot.png", path="runs/"):
    '''
    Archaic function for plotting a heatmap of the weights of a model.
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(len(model.keys())*3, 2)
    vcount = 0
    for count, (name, weights) in enumerate(model.items()):
        if len(weights.shape) == 1:
            weights = weights.reshape(1, -1)
            ax = plt.subplot(gs[vcount, 1])
            vcount += 6
        else:
            ax = plt.subplot(gs[vcount:vcount+4, 0])
        ax.set_title(name)
        if set_abs:  # SET WEIGHTS TO ABS FOR PLOTTING
            weights = np.absolute(weights)
        ax.imshow(weights, cmap='hot', interpolation='nearest')
    plt.suptitle(title)
    if save:
        plt.savefig(save_as, bbox_inches='tight')