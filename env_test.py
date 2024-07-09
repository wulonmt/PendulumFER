import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

import argparse
from datetime import datetime
import Env

if __name__ == "__main__":
    
    n_cpu = 1
    batch_size = 64
    env_name = "PendulumFixPos-v0"
    init_theta = np.pi*3/4
    init_thetadot = 1.0
    #trained_env = GrayScale_env
    trained_env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed = 1, env_kwargs = {"init_theta": init_theta, "init_thetadot": init_thetadot})
    tensorboard_log = "./"

    #trained_env = make_vec_env(GrayScale_env, n_envs=n_cpu,)
    #env = gym.make("highway-fast-v0", render_mode="human")
    model = PPO("MlpPolicy",
                trained_env,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=1,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=1,
                target_kl=0.2,
                ent_coef=0.03,
                vf_coef=0.8,
                tensorboard_log=tensorboard_log)
    time_str = datetime.now().strftime("%Y%m%d%H%M")
    # Train the agent
    model.learn(total_timesteps=int(1e3), tb_log_name=time_str)
    print("log name: ", tensorboard_log + time_str)
    model.save(tensorboard_log + "model")

    model = PPO.load(tensorboard_log + "model")
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name, render_mode="human", init_theta = init_theta, init_thetadot = init_thetadot)
    while True:
        obs, info = env.reset()
        done = truncated = False
        counter = 0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            counter += 1
            if counter > 100:
                break