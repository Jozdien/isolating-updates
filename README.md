# Identifying High-level Structures in RL Agents Through Isolated Updates

This repo contains code for training an RL agent in an environment to optimize a reward, and then training it for a few timesteps on a changed reward.  This is to test one of my theories of interpretability that if you can change one aspect of the training setup after training the model to a sufficient limit, whatever update there are to the weights will correspond to that change, and that this can be used to isolate a pure high-level structure of some property in the network.

For example, I train a PPO agent in the LunarLander-v2 enviroment to optimize landing the lander on a landing pad without crashing.  Then, I train it for a few timesteps on a changed reward that penalizes fuel consumption in the lander. Assuming the theory to be correct, the updates to the weights in these timesteps should correspond to the model's reward function (or some portion thereof).

Of course, there is further work involved in trying to isolate this high-level structure purely.  One approach considered is to subtract these updates from the model prior to this second stage of training and observing whether the model now performs as well on the landing task, while *maximizing* fuel consumption - a priori this may well have no reason to be true if reward is represented by pointers instead of intrinsically, or if there are additional policy changes whose inverse mechanistically do not imply the inverse of the reward function. Further, there could be randomness in the updates during the second phase that could lead to subtracting them hampering the model in other ways.  One potential fix to this is to train the model for a few timesteps more, to see if it regains that capability, and how long it takes to do so.

*On hold while I explore other approaches.*

## Usage

Just leaving some quick info here in case anyone wants to try this out for themselves:

Creating a new run:
```python
python new_run.py
```

Analysing a single run
```python
python analyze.py --runs <Run folder name>
```
This will also store the analysis in the analysis folder.

Comparing stats across multiple runs (adjust the query in the analyze.py code - line 31)
```python
python analyze.py --multiple
```

## To-Do

- Perturb the update signal and see what happens.
- ~~Add analysis functions to graph rewards and any stats of interest.~~
- Try modifying second reward to only be fuel and see what changes, if anything.
- Consider what would verify specific hypotheses about observations.
    - For example, if this doesn't seem to work at all, it could be that the policy is changing as the reward does, in ways that aren't clearly detectable with high-level structural changes (at least, not at the same high-level "objective" would be, which is the hope). Perturbations to the updates (find way to add random noise of variable intensity) for verification?
- Try with different lengths of the second phase of training. Maybe shorter updates are more likely to only update the objective, maybe longer ones average out noise that isn't related to the objective - either way, we'll have to see.