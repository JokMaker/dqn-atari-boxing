# Final Model Training Log — 500k steps (lr=1e-4, batch=64, gamma=0.99)

Reward trend (ep_rew_mean, training rollouts):

| Timesteps | Mean reward | Episode length |
|---|---|---|
| 14k | 3.8 | 7140 |
| 50k | -0.9 | 7140 |
| 100k | -4.0 | 7140 |
| 200k | -2.8 | 7140 |
| 250k | -0.1 | 7140 |
| 300k | +6.2 | 7120 |
| 350k | +11.5 | 7100 |
| 400k | +15.9 | 7030 |
| 450k | +21.9 | 6760 |
| 500k | +24.5 | 6620 |

Note: reward plateaued near -3 between 60k–200k steps, then climbed steadily.
Episode length shortened as the agent improved — winning faster.
Final greedy evaluation: +30.00 ± 16.44 over 5 episodes.
