import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

"""
play_wagner.py
Loads wagner's trained DQN model and plays episodes in ALE/Boxing-v5,
rendering the game live in a window (env_kwargs render_mode="human").
"""

gym.register_envs(ale_py)

ENV_ID = "ALE/Boxing-v5"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "dqn_model_wagner_best.zip")
N_EPISODES = 5


def create_env():
    """Create the Boxing environment with the SAME preprocessing as train_wagner.py."""
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=0,
        env_kwargs={"frameskip": 1, "render_mode": "human"},
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def play():
    """Load the trained model and play episodes with greedy action selection."""
    env = create_env()
    model = DQN.load(MODEL_PATH, env=env)

    for episode in range(1, N_EPISODES + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # deterministic=True = GreedyQPolicy: always pick argmax Q(s, a)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

        print(f"Episode {episode}: total reward = {total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    play()
