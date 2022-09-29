import gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from wrappers import RewardWrapper
import utils

BASE_ENV = 'LunarLander-v2'

POLICY = 'MlpPolicy'

FIRST_TRAIN_TIMESTEPS = 1000 # 25000 recommended
SECOND_TRAIN_TIMESTEPS = 1000

metadata = {
    'BASE_ENV': BASE_ENV,
    'POLICY': POLICY,
    'FIRST_TRAIN_TIMESTEPS': FIRST_TRAIN_TIMESTEPS,
    'SECOND_TRAIN_TIMESTEPS': SECOND_TRAIN_TIMESTEPS,
}

rewards_dict = {
    'pre_update': {
        'total_reward': [],
        'fuel_reward': [],
    },
    'post_update': {
        'total_reward': [],
        'fuel_reward': [],
    },
    'sub_update': {
        'total_reward': [],
        'fuel_reward': [],
    }
}

dir_name = utils.mkdir_timestamped()
log_path = dir_name + "log/"

new_logger = configure(log_path, ['stdout', 'csv', 'json'])

base_env = gym.make(BASE_ENV, continuous=True)  # Using the continuous LunarLander environment to broaden fuel consumption action space.
no_fuel_env = RewardWrapper(base_env, fuel=False)
fuel_env = RewardWrapper(base_env, fuel=True)

model = PPO(POLICY, no_fuel_env, verbose=1)
init_weights = utils.true_copy(utils.get_weights(model))
model.save(dir_name + 'init_weights_model')

model.set_logger(new_logger)
model.learn(total_timesteps=FIRST_TRAIN_TIMESTEPS)

model.set_env(fuel_env)
pre_update_weights = utils.true_copy(utils.get_weights(model))
model.save(dir_name + 'pre_update_weights_model')

obs = fuel_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = fuel_env.step(action)
    rewards_dict['pre_update']['total_reward'].append(reward)
    rewards_dict['pre_update']['fuel_reward'].append(info['fuel_reward'])
    # fuel_env.render()
    if done:
      obs = fuel_env.reset()

model.learn(total_timesteps=SECOND_TRAIN_TIMESTEPS)
post_update_weights = utils.true_copy(utils.get_weights(model))
model.save(dir_name + 'post_update_weights_model')

obs = fuel_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = fuel_env.step(action)
    rewards_dict['post_update']['total_reward'].append(reward)
    rewards_dict['post_update']['fuel_reward'].append(info['fuel_reward'])
    # fuel_env.render()
    if done:
      obs = fuel_env.reset()

updates = utils.weight_diff(post_update_weights, pre_update_weights)

sub_update_weights = utils.weight_diff(pre_update_weights, updates)
model.set_parameters(utils.to_dict('policy', sub_update_weights), exact_match=False)
model.save(dir_name + 'sub_update_weights_model')

utils.save_metadata(metadata, path=dir_name)
utils.save_weights(utils.saveable(init_weights), path=dir_name, name="init_weights.json")
utils.save_weights(utils.saveable(pre_update_weights), path=dir_name, name="pre_update_weights.json")
utils.save_weights(utils.saveable(post_update_weights), path=dir_name, name="post_update_weights.json")
utils.save_weights(utils.saveable(updates), path=dir_name, name="updates.json")
utils.save_weights(utils.saveable(sub_update_weights), path=dir_name, name="sub_update_weights.json")

utils.heatmap_sep(updates, title="Updates", show=False, save=True, save_as_prefix="updates.png", path=dir_name+"plots/")

obs = fuel_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = fuel_env.step(action)
    rewards_dict['sub_update']['total_reward'].append(reward)
    rewards_dict['sub_update']['fuel_reward'].append(info['fuel_reward'])
    # fuel_env.render()
    if done:
      obs = fuel_env.reset()

utils.save_to_json(rewards_dict, path=dir_name, name="rewards.json")