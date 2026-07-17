import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import csv
import os
from stable_baselines3.common.evaluation import evaluate_policy

"""
train_wagner.py
Trains a DQN agent on ALE/Boxing-v5 using Stable Baselines3.
Same env/architecture as the group's train.py, but this is wagner's own
copy: own hyperparameter sweep, own experiment log, own saved models.
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
    total_timesteps=150_000,
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
    model.save(f"models/dqn_model_wagner_{run_name}.zip")
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
            run_name, policy, params["learning_rate"], params["gamma"],
            params["batch_size"], params["exploration_initial_eps"],
            params["exploration_final_eps"], params["exploration_fraction"],
            round(mean_reward, 2), round(std_reward, 2),
        ])


if __name__ == "__main__":
    # wagner's sweep: 10 required hyperparameter experiments (CnnPolicy)
    # + 1 extra MlpPolicy run to document the CNN-vs-MLP comparison.
    MEMBER = "wagner"
    TIMESTEPS_PER_RUN = 150_000  # lower to 100_000 if short on time/compute

    EXPERIMENTS = [
        dict(run_name="exp01_baseline", policy="CnnPolicy", params=dict(
            learning_rate=1e-4, gamma=0.99, batch_size=32,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp02_lr_mid_high", policy="CnnPolicy", params=dict(
            learning_rate=5e-4, gamma=0.99, batch_size=32,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp03_lr_very_low", policy="CnnPolicy", params=dict(
            learning_rate=5e-6, gamma=0.99, batch_size=32,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp04_gamma_high", policy="CnnPolicy", params=dict(
            learning_rate=1e-4, gamma=0.999, batch_size=32,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp05_gamma_mid", policy="CnnPolicy", params=dict(
            learning_rate=1e-4, gamma=0.95, batch_size=32,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp06_batch_very_high", policy="CnnPolicy", params=dict(
            learning_rate=1e-4, gamma=0.99, batch_size=128,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp07_batch_very_low", policy="CnnPolicy", params=dict(
            learning_rate=1e-4, gamma=0.99, batch_size=8,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp08_eps_start_low", policy="CnnPolicy", params=dict(
            learning_rate=1e-4, gamma=0.99, batch_size=32,
            exploration_initial_eps=0.5, exploration_final_eps=0.05, exploration_fraction=0.1)),
        dict(run_name="exp09_eps_end_very_low", policy="CnnPolicy", params=dict(
            learning_rate=1e-4, gamma=0.99, batch_size=32,
            exploration_initial_eps=1.0, exploration_final_eps=0.01, exploration_fraction=0.2)),
        dict(run_name="exp10_tuned_combo", policy="CnnPolicy", params=dict(
            # Edit this AFTER exp01-09 finish: combine whichever individual
            # changes helped most (e.g. best lr + best gamma + best batch_size).
            learning_rate=5e-4, gamma=0.999, batch_size=128,
            exploration_initial_eps=1.0, exploration_final_eps=0.01, exploration_fraction=0.2)),
    ]

    # Extra (not one of the required 10): MLP vs CNN architecture comparison,
    # same hyperparameters as the baseline, only the policy network differs.
    MLP_COMPARISON = dict(run_name="exp_mlp_vs_cnn", policy="MlpPolicy", params=dict(
        learning_rate=1e-4, gamma=0.99, batch_size=32,
        exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.1))

    for exp in EXPERIMENTS + [MLP_COMPARISON]:
        if already_run(MEMBER, exp["run_name"]):
            print(f"\n=== Skipping {exp['run_name']} (already logged) ===")
            continue
        print(f"\n=== Running {exp['run_name']} ({exp['policy']}) ===")
        model = train_agent(
            policy=exp["policy"],
            total_timesteps=TIMESTEPS_PER_RUN,
            run_name=exp["run_name"],
            **exp["params"],
        )
        mean_reward, std_reward = evaluate_agent(model)
        log_experiment(MEMBER, exp["run_name"], exp["policy"], exp["params"], mean_reward, std_reward)
        print(f"{exp['run_name']}: mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
