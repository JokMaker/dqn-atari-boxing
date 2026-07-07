import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import csv
import os
from stable_baselines3.common.evaluation import evaluate_policy

"""
train.py
Trains a DQN agent on ALE/Boxing-v5 using Stable Baselines3.
Saves the trained model as dqn_model.zip.
"""

gym.register_envs(ale_py)

ENV_ID = "ALE/Boxing-v5"


def create_env(n_envs=1, seed=42):
    """Create the Boxing environment with standard Atari preprocessing."""
    env = make_atari_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"frameskip": 1},
    )
    env = VecFrameStack(env, n_stack=4)
    return env

def train_agent(
    policy="CnnPolicy",
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    exploration_fraction=0.1,
    total_timesteps=100_000,
    run_name="baseline",
):
    """Train a DQN agent with the given hyperparameters and save it."""
    env = create_env()

    model = DQN(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        buffer_size=100_000,
        learning_starts=10_000,
        target_update_interval=1_000,
        train_freq=4,
        verbose=1,
        tensorboard_log="logs/tensorboard/",
    )

    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)
    model.save(f"models/dqn_model_{run_name}.zip")
    env.close()
    return model


def evaluate_agent(model, n_episodes=5):
    """Evaluate a trained model and return mean reward and episode length."""
    eval_env = create_env(seed=100)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    eval_env.close()
    return mean_reward, std_reward


def log_experiment(member, run_name, params, mean_reward, std_reward):
    """Append experiment results to a CSV file for the README table."""
    filepath = f"experiments/{member}_experiments.csv"
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_name", "lr", "gamma", "batch_size",
                "eps_start", "eps_end", "eps_fraction",
                "mean_reward", "std_reward",
            ])
        writer.writerow([
            run_name, params["learning_rate"], params["gamma"],
            params["batch_size"], params["exploration_initial_eps"],
            params["exploration_final_eps"], params["exploration_fraction"],
            round(mean_reward, 2), round(std_reward, 2),
        ])


if __name__ == "__main__":
    # Experiment configuration — edit these values for each run
    MEMBER = "jok"
    RUN_NAME = "exp01_baseline"
    PARAMS = {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.1,
    }

    model = train_agent(
        policy="CnnPolicy",
        total_timesteps=100_000,
        run_name=RUN_NAME,
        **PARAMS,
    )

    mean_reward, std_reward = evaluate_agent(model)
    log_experiment(MEMBER, RUN_NAME, PARAMS, mean_reward, std_reward)
    print(f"{RUN_NAME}: mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")