def create_env():
    """Create the Boxing environment with the SAME preprocessing as train.py."""
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