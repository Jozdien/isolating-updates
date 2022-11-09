import gym
from gym.wrappers import TimeLimit, StepAPICompatibility
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from wrappers import RewardWrapper
import utils

BASE_ENV = 'LunarLander-v2'
FIRST_WRAPPER = 'no_fuel_env'  # env wrapper used for primary training
SECOND_WRAPPER = 'scaled_fuel_env'  # env wrapper using for training to subtract updates of
TEST_ENV = 'fuel_env'  # env wrapper used for testing model performance

POLICY = 'MlpPolicy'

FIRST_TRAIN_TIMESTEPS = 250000
SECOND_TRAIN_TIMESTEPS = 90000

metadata = {
    'BASE_ENV': BASE_ENV,
    'FIRST_WRAPPER': FIRST_WRAPPER,
    'SECOND_WRAPPER': SECOND_WRAPPER,
    'TEST_ENV': TEST_ENV,
    'POLICY': POLICY,
    'FIRST_TRAIN_TIMESTEPS': FIRST_TRAIN_TIMESTEPS,
    'SECOND_TRAIN_TIMESTEPS': SECOND_TRAIN_TIMESTEPS,
}

rewards_dict = {
    'pre_update': { 'total_reward': [], 'fuel_reward': [] },
    'post_update': { 'total_reward': [], 'fuel_reward': [] },
    'sub_update': { 'total_reward': [], 'fuel_reward': [] }
}

# create base environment for wrappers
base_env = gym.make(BASE_ENV)  # using the continuous LunarLander environment to broaden fuel consumption action space
wrappers = {
    'no_fuel_env': TimeLimit(StepAPICompatibility(RewardWrapper(base_env, include_fuel=False, only_fuel=False, scale_fuel=1))),  # env with no penalty for fuel consumption
    'fuel_env': TimeLimit(StepAPICompatibility(RewardWrapper(base_env, include_fuel=True, only_fuel=False, scale_fuel=1))),  # env with penalty for fuel consumption
    'only_fuel_env': TimeLimit(StepAPICompatibility(RewardWrapper(base_env, include_fuel=True, only_fuel=True, scale_fuel=1))),  # fuel penalty as only reward
    'scaled_fuel_env': TimeLimit(StepAPICompatibility(RewardWrapper(base_env, include_fuel=True, only_fuel=False, scale_fuel=5))),  # fuel penalty scaled by 5
    'only_scaled_fuel_env': TimeLimit(StepAPICompatibility(RewardWrapper(base_env, include_fuel=True, only_fuel=True, scale_fuel=5)))  # penalty as only reward, scaled by 5
}

FIRST_TRAIN_ENV = wrappers[FIRST_WRAPPER]
SECOND_TRAIN_ENV = wrappers[SECOND_WRAPPER]
TEST_ENV = wrappers[TEST_ENV]

dir_name = utils.mkdir_timestamped()  # create a new directory for this run
log_path = dir_name + "log/"  # directory for the files saved by logger

new_logger = configure(log_path, ['stdout', 'json', 'csv'])  # custom logger to save as viewable JSON

model = PPO(POLICY, FIRST_TRAIN_ENV, verbose=1)
init_weights = utils.true_copy(utils.get_weights(model))
model.save(dir_name + 'init_weights_model')

model.set_logger(new_logger)
model.learn(total_timesteps=FIRST_TRAIN_TIMESTEPS)

model.set_env(SECOND_TRAIN_ENV)
pre_update_weights = utils.true_copy(utils.get_weights(model))
model.save(dir_name + 'pre_update_weights_model')

utils.test_model(TEST_ENV, model, rewards_dict, 'pre_update', render=False, num_episodes=5000)

model.learn(total_timesteps=SECOND_TRAIN_TIMESTEPS)
post_update_weights = utils.true_copy(utils.get_weights(model))
model.save(dir_name + 'post_update_weights_model')

utils.test_model(TEST_ENV, model, rewards_dict, 'post_update', render=False, num_episodes=5000)

updates = utils.weight_diff(post_update_weights, pre_update_weights)

sub_update_weights = utils.weight_diff(pre_update_weights, updates)
model.set_parameters(utils.to_dict('policy', sub_update_weights), exact_match=False)

# TESTING ADDING THESE TWO LINES
# if subtracted model can re-learn capabilities and still maximizes fuel, then the update signal contains objective
# if it can't - it might still just have a lot of interference, but it might also be that the update signal is just noise
# to separate from the other runs, SECOND_TRAIN_TIMESTEPS is set to 90k
model.set_env(FIRST_TRAIN_ENV)
model.learn(total_timesteps=100000)

model.save(dir_name + 'sub_update_weights_model')

utils.test_model(TEST_ENV, model, rewards_dict, 'sub_update', render=False, num_episodes=5000)

utils.save_to_json(rewards_dict, path=dir_name, name="rewards.json")

utils.save_metadata(metadata, path=dir_name)
utils.update_rundir(name=dir_name, metadata=metadata)
utils.save_weights(utils.saveable(init_weights), path=dir_name, name="init_weights.json")
utils.save_weights(utils.saveable(pre_update_weights), path=dir_name, name="pre_update_weights.json")
utils.save_weights(utils.saveable(post_update_weights), path=dir_name, name="post_update_weights.json")
utils.save_weights(utils.saveable(updates), path=dir_name, name="updates.json")
utils.save_weights(utils.saveable(sub_update_weights), path=dir_name, name="sub_update_weights.json")

utils.heatmap_sep(updates, title="Updates", show=False, save=True, save_as_prefix="updates.png", path=dir_name+"plots/")