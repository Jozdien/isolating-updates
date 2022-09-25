import torch
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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

def mkdir_timestamped(path="plots/"):
    '''
    Creates a directory with the current date and time.
    '''
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = os.path.join(os.getcwd(), path, time)
    os.mkdir(name)
    return time

def heatmap_sep(model, title="Plot", set_abs=True, show=True, save=False, save_as_prefix="plot", path="plots/"):
    '''
    Plots separate heatmaps of every weight tensor in a model.
    '''
    if save_as_prefix.endswith(".png"):
        save_as_prefix = save_as_prefix[:-4]
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

def heatmap(model, title="Plot", set_abs=True, save=False, save_as="plot.png", path="plots/"):
    '''
    Archaic function for plotting a heatmap of the weights of a model.
    '''
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