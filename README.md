## MLP vs CNN Policy Comparison

Under identical hyperparameters (`lr=1e-4`, `gamma=0.99`, `batch_size=32`, epsilon 1.0 → 0.05 over the first 10% of training), we compared the two required policy architectures on ALE/Boxing-v5:

| Policy | Mean Reward | Std |
|---|---|---|
| `CnnPolicy` | -10.8 | ±3.31 |
| `MlpPolicy` | -34.6 | ±0.49 |

CNN outperforms MLP by 24 points at otherwise identical settings.

**Why:** Boxing's observation is 4 stacked 84x84 grayscale frames. `MlpPolicy` flattens this into one long vector before any hidden layer, destroying all spatial structure — it has no way to represent what the boxer sprites look like or where they are relative to each other. `CnnPolicy`'s convolutional filters instead preserve spatial relationships between pixels as they slide across the image, which is exactly the information Boxing decisions depend on (where the opponent is, which direction to move or punch).

The MLP's much smaller variance (±0.49 vs CNN's ±3.31) reinforces this: it isn't failing noisily on some episodes and doing better on others — it consistently converged to one fixed, ineffective behavior every episode, indicating it never learned any meaningful visual features at all.

**Decision:** Based on this comparison, all hyperparameter tuning experiments in this project use `CnnPolicy`.
