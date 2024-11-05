import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import A2C, DQN, PPO, SAC

import os

class FeatureExtractor2048(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 6144):
        super().__init__(observation_space, features_dim)
        self.adjacent_pairs = [
            ((i, j), (i + di, j + dj))
            for i in range(4) for j in range(4)
            for di, dj in [(1, 0), (0, 1)]
            if 0 <= i + di < 4 and 0 <= j + dj < 4
        ]
    
    def unstack(self, layered):
        # Replace NumPy with PyTorch operations
        representation = torch.arange(16, device=layered.device, dtype=torch.float32)
        flat = torch.sum(layered * representation.view(-1, 1, 1), dim=1)
        flat = flat.view(layered.size(0), 4, 4).to(torch.int)
        return flat
    
    def forward(self, obs):
        # Use torch.zeros directly to avoid NumPy
        batch_size = obs.size(0)
        encoded_obs = torch.zeros((batch_size, 24, 16, 16), device=obs.device, dtype=torch.float32)

        observation = self.unstack(obs)

        # Efficiently encode adjacency features without explicit loops
        for idx, (pos1, pos2) in enumerate(self.adjacent_pairs):
            power1 = observation[:, pos1[0], pos1[1]].to(torch.long)
            power2 = observation[:, pos2[0], pos2[1]].to(torch.long)
            # Use advanced indexing to set values in encoded_obs directly
            encoded_obs[torch.arange(batch_size), idx, power1, power2] = 1
        
        # Flatten for feature dimension
        encoded_obs = encoded_obs.view(batch_size, -1)
        return encoded_obs

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": f"{len(os.listdir('models'))}",

    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": f"models/{len(os.listdir('models'))}",

    "epoch_num": 10000,
    "timesteps_per_epoch": 1000,
    "eval_episode_num": 10,
    "learning_rate": 1e-4,

    "policy_kwargs": dict(
        features_extractor_class=FeatureExtractor2048,
        net_arch=[dict(pi=[64, 64, 64, 64], vf=[64, 64])]
    ),
}

def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
        
    return avg_score, avg_highest

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):

        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                verbose=2,
            ),
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        
        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print()
        wandb.log(
            {"avg_highest": avg_highest,
             "avg_score": avg_score}
        )
        

        ### Save best model
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="RL_hw3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"]
    )

    # Create training environment 
    num_train_envs = 10
    train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])

    # Create evaluation environment 
    eval_env = DummyVecEnv([make_env])  

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env,
        verbose=1,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=my_config["policy_kwargs"]
    )

    print(model.policy)

    train(eval_env, model, my_config)