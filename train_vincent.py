import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import csv
import os
from stable_baselines3.common.evaluation import evaluate_policy
import sys

"""
train_vincent.py
Trains a DQN agent on ALE/Boxing-v5 using Stable Baselines3.
Vincent's own hyperparameter sweep and experiment log.
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

    model.learn(total_timesteps=total_timesteps,
                tb_log_name=f"vincent_{run_name}")
    model.save(f"models/dqn_model_vincent_{run_name}.zip")
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


def already_run(member, run_name):
    """Check whether run_name has already been logged (lets a crashed sweep resume safely)."""
    filepath = f"experiments/{member}_experiments.csv"
    if not os.path.isfile(filepath):
        return False
    with open(filepath, newline="") as f:
        return run_name in {row["run_name"] for row in csv.DictReader(f)}


def log_experiment(member, run_name, policy, params, mean_reward, std_reward):
    """Append experiment results to a CSV file for the README table."""
    filepath = f"experiments/{member}_experiments.csv"
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_name", "policy", "lr", "gamma", "batch_size",
                "eps_start", "eps_end", "eps_fraction",
                "mean_reward", "std_reward",
            ])
        writer.writerow([
            run_name,
            policy,
            params["lr"],
            params["gamma"],
            params["batch_size"],
            params["eps_start"],
            params["eps_end"],
            params["eps_fraction"],
            mean_reward,
            std_reward,
        ])


def main():
    """Run a hyperparameter sweep with custom experiments."""
    member = "vincent"

    # Prevent multiple instances from running simultaneously
    lock_file = f"experiments/.{member}_training.lock"
    if os.path.exists(lock_file):
        print(
            f"ERROR: Another instance of {member} training is already running.")
        print(
            f"Delete {lock_file} if you're sure no other process is running.")
        sys.exit(1)

    try:
        # Create lock file
        os.makedirs("experiments", exist_ok=True)
        with open(lock_file, "w") as f:
            f.write("training in progress\n")

        # Define your hyperparameter sweep here
        experiments = [
            {
                "run_name": "exp04_lr_mid",
                "policy": "CnnPolicy",
                "lr": 2e-4,
                "gamma": 0.99,
                "batch_size": 32,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp05_gamma_mid",
                "policy": "CnnPolicy",
                "lr": 1e-4,
                "gamma": 0.97,
                "batch_size": 32,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp06_gamma_low_mid",
                "policy": "CnnPolicy",
                "lr": 1e-4,
                "gamma": 0.92,
                "batch_size": 32,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp07_batch_mid",
                "policy": "CnnPolicy",
                "lr": 1e-4,
                "gamma": 0.99,
                "batch_size": 48,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp08_eps_start_mid",
                "policy": "CnnPolicy",
                "lr": 1e-4,
                "gamma": 0.99,
                "batch_size": 32,
                "eps_start": 0.8,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp09_eps_end_low",
                "policy": "CnnPolicy",
                "lr": 1e-4,
                "gamma": 0.99,
                "batch_size": 32,
                "eps_start": 1.0,
                "eps_end": 0.03,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp10_eps_fraction_mid",
                "policy": "CnnPolicy",
                "lr": 1e-4,
                "gamma": 0.99,
                "batch_size": 32,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.15,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp11_balanced_tuning",
                "policy": "CnnPolicy",
                "lr": 2e-4,
                "gamma": 0.97,
                "batch_size": 48,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp12_aggressive_batch",
                "policy": "CnnPolicy",
                "lr": 1e-4,
                "gamma": 0.99,
                "batch_size": 96,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
            {
                "run_name": "exp13_conservative_lr",
                "policy": "CnnPolicy",
                "lr": 7e-5,
                "gamma": 0.99,
                "batch_size": 32,
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_fraction": 0.1,
                "total_timesteps": 100_000,
            },
        ]

        for exp in experiments:
            run_name = exp["run_name"]

            # Skip if already run (allows resuming after crashes)
            if already_run(member, run_name):
                print(f"Skipping {run_name} (already run)")
                continue

            print(f"\n{'='*60}")
            print(f"Running: {run_name}")
            print(f"{'='*60}")

            # Train
            model = train_agent(
                policy=exp["policy"],
                learning_rate=exp["lr"],
                gamma=exp["gamma"],
                batch_size=exp["batch_size"],
                exploration_initial_eps=exp["eps_start"],
                exploration_final_eps=exp["eps_end"],
                exploration_fraction=exp["eps_fraction"],
                total_timesteps=exp["total_timesteps"],
                run_name=run_name,
            )

            # Evaluate
            mean_reward, std_reward = evaluate_agent(model, n_episodes=5)
            print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

            # Log
            params = {
                "lr": exp["lr"],
                "gamma": exp["gamma"],
                "batch_size": exp["batch_size"],
                "eps_start": exp["eps_start"],
                "eps_end": exp["eps_end"],
                "eps_fraction": exp["eps_fraction"],
            }
            log_experiment(member, run_name,
                           exp["policy"], params, mean_reward, std_reward)

    finally:
        # Remove lock file
        lock_file = f"experiments/.{member}_training.lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)
            print(f"\nTraining complete. Lock file removed.")


if __name__ == "__main__":
    main()
