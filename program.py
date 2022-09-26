import gym
from stable_baselines3 import PPO

from wrapper import DualRewardWrapper
import utils

base_env = gym.make('LunarLander-v2')
custom_env = DualRewardWrapper(base_env, flag=True)

model = PPO("MlpPolicy", base_env, verbose=1)
model.learn(total_timesteps=1000) # 25000 recommended

model.set_env(custom_env)

weights = utils.true_copy(utils.get_weights(model))
model.learn(total_timesteps=1000)
new_weights = utils.true_copy(utils.get_weights(model))

updates = utils.weight_diff(weights, new_weights)

dir_name = utils.mkdir_timestamped()
utils.heatmap_sep(updates, title="Original Weights", show=False, save=True, save_as_prefix="original_weights.png", path="plots/{}/".format(dir_name))

# obs = env.reset()
# for i in range(500):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = custom_env.step(action)
#     done = terminated or truncated
#     custom_env.render()
#     if done:
#       obs = custom_env.reset()