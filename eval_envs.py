import gymnasium as gym
import argparse
import Env
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_model", help="model to be logged", type=str)
args = parser.parse_args()
INIT_POS = [{"init_theta": np.pi*3/4, "init_thetadot": 1}, {"init_theta": -np.pi*3/4, "init_thetadot": 1}, {"init_theta": np.pi/2, "init_thetadot": 1}, {"init_theta": -np.pi/2, "init_thetadot": 1}, {"init_theta": np.pi, "init_thetadot": 1}]

def round_floats(value):
    return round(value, 2) if isinstance(value, float) else value

if __name__ == "__main__":
    env_name = "PendulumFixPos-v0"
    folder_path = [f.path for f in os.scandir(args.log_model) if f.is_dir()]
    
    # Open a CSV file to write the results
    with open(args.log_model + '/evaluation_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['agent', 'init_theta', 'init_thetadot', 'reward_mean', 'reward_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        eval_env_mean = {0: [], 1: [], 2: [], 3: [], 4: []}
        eval_env_std = {0: [], 1: [], 2: [], 3: [], 4: []}

        for path in folder_path:
            model = PPO.load(path+"/model")
            dir_name = path.split("\\")[-1]
            print(f"{dir_name = }")
            for index, para_dict in enumerate(INIT_POS):
                env = gym.make(env_name, render_mode="rgb_array", **para_dict)
                env.reset()
                reward_mean, reward_std = evaluate_policy(model, env)
                print(f"{para_dict = }, {reward_mean = }, {reward_std = }")
                
                # Write the results to the CSV file
                writer.writerow({
                    'agent': dir_name.split("_")[0],
                    'init_theta': round_floats(para_dict['init_theta']),
                    'init_thetadot': round_floats(para_dict['init_thetadot']),
                    'reward_mean': round_floats(reward_mean),
                    'reward_std': round_floats(reward_std)
                })
        AVG = lambda x: sum(x)/len(x)
        for env, (mean, std) in enumerate(zip(eval_env_mean.values(), eval_env_std.values())):
            print(f"env {env}, reward {AVG(mean)}, std {AVG(std)}")

    print("Results have been saved to evaluation_results.csv")