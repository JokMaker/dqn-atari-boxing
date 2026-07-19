# Deep Q-Learning — Atari Boxing (ALE/Boxing-v5)

Formative 3 | Machine Learning Techniques 1 | African Leadership University

## Group Members

| Member | Responsibilities |
|--------|------------------|
| Jok John Maker Kur | Repository setup, train.py, shared Colab notebook, 10 hyperparameter experiments, final 500k-step training run, README assembly |
| Vincent | 10+ hyperparameter experiments, play.py testing |
| Wagner | 10 hyperparameter experiments, MLP vs CNN policy comparison, gameplay video, assignment sheet (PDF) |

## Environment

**ALE/Boxing-v5** — the agent controls a boxer and scores +1 for each punch landed on the opponent, -1 for each punch received. Episode scores range from -100 to +100 (100 points is a knockout). We chose Boxing because its dense, immediate reward signal suits DQN well: every short experiment run produces measurable reward differences, making hyperparameter comparisons meaningful.

The v5 environment uses sticky actions (`repeat_action_probability=0.25`): 25% of the time the environment repeats the previous action instead of the new one. This injects stochasticity so agents cannot memorize fixed action sequences — and explains why even our greedy-policy agent shows score variance between episodes.

## Final Result

| Model | Training steps | Evaluation (5 episodes, greedy) |
|-------|---------------|----------------------------------|
| Baseline config | 100,000 | -1.00 ± 3.41 |
| Best tuned config (exp10) | 100,000 | +7.40 ± 7.76 |
| **Final model (dqn_model.zip)** | **500,000** | **+30.00 ± 16.44** |

Final configuration: `lr=1e-4, gamma=0.99, batch_size=64, eps 1.0 → 0.05 over 10% of training`.

The training curve for the final run was nearly flat around -3 between 60k and 200k steps, then climbed steadily to a training average of +24.5 by 500k steps — evidence that our 100k-step experiments were measuring early learning dynamics, and that the tuned configuration had substantial headroom with more training time.

## Project Structure

```
train.py               # Canonical training script — DQN agent, experiment logging
play.py                # Loads models/dqn_model.zip, plays with rendering (greedy policy)
train_vincent.py       # Vincent's experiment sweep (variant of train.py)
train_wagner.py        # Wagner's experiment sweep (variant of train.py)
notebooks/             # Shared Colab notebook used for GPU training
models/dqn_model.zip   # Final trained model (500k steps)
experiments/           # Each member's experiment results (CSV)
logs/                  # Training logs
docs/                  # Gameplay video + assignment sheet (PDF)
```

`train.py` is the canonical script; the per-member variants document each member's individual sweep. All experiments used identical environment preprocessing, seeds, and fixed DQN settings so that runs differ only in the hyperparameters under test.

## Setup

```
pip install -r requirements.txt
python train.py     # trains a DQN agent
python play.py      # loads models/dqn_model.zip and plays 3 rendered episodes
```

## Preprocessing and Fixed Settings

All runs used the standard Atari DQN pipeline: grayscale conversion, resize to 84x84, reward clipping to {-1, 0, +1}, frame skip of 4 (environment `frameskip=1` so only the wrapper skip applies — avoiding double frame-skipping with v5's built-in skip), and a stack of 4 consecutive frames so the network can perceive motion. Fixed DQN settings across all experiments: `buffer_size=100k`, `learning_starts=10k`, `target_update_interval=1k`, `train_freq=4`. Training seed 42, evaluation seed 100, evaluation over 5 episodes with `deterministic=True` (greedy policy).

## Policy Comparison: MLP vs CNN

We compared both policies at identical baseline hyperparameters (100k steps):

| Policy | Evaluation reward |
|--------|-------------------|
| CnnPolicy | -10.8 ± 3.31 |
| MlpPolicy | -34.6 ± 0.49 |

The MLP performed at random-play level and its near-zero variance shows it consistently learned nothing. The reason is structural: the observation is 4 stacked 84x84 frames. An MLP flattens this into a ~28,000-value vector, destroying all spatial relationships — the network cannot learn what the boxer sprites are or where they are relative to each other. A CNN's convolutional filters slide across the image and learn local spatial patterns (edges, sprites, relative positions) with far fewer parameters through weight sharing. The relative position of the two boxers is precisely the information Boxing decisions depend on. **All experiments therefore used CnnPolicy.**

## Hyperparameter Tuning Results

Each member ran their experiments against a personal baseline because identical seeds do not guarantee identical results across GPU sessions (CUDA operations are non-deterministic, and small divergences compound over 100k steps — DQN is known to have high run-to-run variance). Within-member comparisons are controlled; cross-member comparisons are indicative.

### Jok — 10 Experiments

| # | Run | lr | gamma | batch | eps_start | eps_end | eps_frac | Mean reward | Noted behaviour |
|---|-----|----|----|----|----|----|----|----|----|
| 1 | baseline | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | -1.00 ± 3.41 | Reference. Reward improved steadily from -19 to -11 during training; greedy evaluation reached approximately even score. Stable learning, no divergence. |
| 2 | lr_high | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | -33.00 ± 0.00 | 10x higher learning rate degraded performance: reward fell to -29 and plateaued, never recovering. Oversized gradient steps caused Q-value estimates to overshoot, converging to a poor policy. |
| 3 | lr_low | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | -6.40 ± 8.31 | 10x lower learning rate: stable but very slow learning — recovered only ~2 points vs baseline's ~8 over the same steps. High variance indicates an under-converged, inconsistent policy. |
| 4 | gamma_low | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.1 | -2.20 ± 4.75 | Minimal impact. Boxing's dense, immediate reward signal means the agent doesn't rely heavily on long-horizon planning. |
| 5 | gamma_extreme | 1e-4 | 0.50 | 32 | 1.0 | 0.05 | 0.1 | -1.20 ± 6.91 | Even a ~2-step planning horizon performed at baseline level, though variance increased. Confirms gamma insensitivity for this reward structure. |
| 6 | batch_high | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | -1.20 ± 6.21 | Visibly smoother training curve (dip of -10 vs baseline's -19) from lower gradient noise, but unchanged final evaluation at ~10% slower training. Stability benefit without performance benefit at this budget. |
| 7 | batch_low | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.1 | -51.20 ± 3.19 | Severe failure — worst result of all experiments. Training plateaued at -18 by 28k steps with no recovery; loss varied across three orders of magnitude. Gradient estimates too noisy for stable Q-learning; the agent converged consistently to a poor policy. |
| 8 | eps_end_high | 1e-4 | 0.99 | 32 | 1.0 | 0.20 | 0.1 | -4.40 ± 1.02 | Persistent 20% random actions mildly degraded the learned policy: one in five replay-buffer transitions came from random moves, polluting the learning data. Lowest variance of all runs — broader exploration produced a consistent but weaker policy. |
| 9 | eps_slow_decay | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.5 | -3.00 ± 3.74 | Stretching decay over 50% of training delayed learning rather than improving it. Boxing's dense rewards don't require prolonged exploration to discover good behaviour. |
| 10 | tuned_combo | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | **+7.40 ± 7.76** | Combining the best-performing values achieved the only positive score in this set. The larger batch's gradient stability compounded with the optimal learning rate; the curve was still climbing at 100k steps — motivating the 500k final run. |

### Wagner — 10 Experiments + MLP Comparison

Baseline reference: -10.8 ± 3.31.

| # | Run | lr | gamma | batch | eps_start | eps_end | eps_frac | Mean reward | Noted behaviour |
|---|-----|----|----|----|----|----|----|----|----|
| 1 | baseline | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | -10.8 ± 3.31 | Reference for this member's runs. |
| 2 | lr_mid_high | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | -25.0 ± 4.98 | Degradation already visible at 5e-4 — locates where the learning-rate cliff begins, between the group's good 1e-4/2e-4 and diverged 1e-3. |
| 3 | lr_very_low | 5e-6 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | -74.6 ± 24.96 | Catastrophically slow: updates too small to learn anything within budget. Huge variance — an essentially unformed policy. |
| 4 | gamma_high | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.1 | -16.6 ± 12.85 | Very long horizon slightly worse than baseline in this run. |
| 5 | gamma_mid | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | **+4.2 ± 6.01** | Best single change in this set — a mid-length horizon outperformed both extremes here. |
| 6 | batch_very_high | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.1 | -4.8 ± 7.93 | Improved on baseline — consistent with the group-wide pattern that larger batches stabilize learning. |
| 7 | batch_very_low | 1e-4 | 0.99 | 8 | 1.0 | 0.05 | 0.1 | -33.4 ± 2.58 | Collapsed, echoing the batch=16 failure in Jok's set: very small batches are unreliable. |
| 8 | eps_start_low | 1e-4 | 0.99 | 32 | 0.5 | 0.05 | 0.1 | -16.2 ± 4.02 | Starting with less exploration hurt — early random coverage of the state space matters. |
| 9 | eps_end_very_low | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.2 | -34.6 ± 11.6 | Cutting exploration nearly to zero degraded performance notably in this run. |
| 10 | tuned_combo | 1e-4 | 0.95 | 128 | 1.0 | 0.05 | 0.1 | +2.6 ± 3.2 | Combining this member's two winners (gamma=0.95, batch=128) produced a positive score. |
| — | mlp_vs_cnn | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | -34.6 ± 0.49 | MlpPolicy at baseline settings: random-level play with near-zero variance — consistently learned nothing from flattened pixels. |

### Vincent — Experiments

| # | Run | lr | gamma | batch | eps_start | eps_end | eps_frac | Mean reward | Noted behaviour |
|---|-----|----|----|----|----|----|----|----|----|
| 1 | baseline | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | (see CSV) | Reference for this member's runs. |
| 2 | lr_mid | 2e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | +2.4 ± 4.36 | Extends the learning-rate curve: 2e-4 still performs well, so the safe zone spans roughly 1e-4 to 2e-4. |
| 3 | gamma_high | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.1 | +5.4 ± 10.33 | Best run in this set — notably, the same config scored -16.6 in Wagner's session, illustrating DQN's run-to-run variance. |
| 4 | gamma_mid | 1e-4 | 0.97 | 32 | 1.0 | 0.05 | 0.1 | -6.2 ± 8.06 | Mid-range gamma below baseline in this run. |
| 5 | gamma_low | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | -0.2 ± 5.84 | Roughly baseline-level, directionally consistent with Wagner's positive gamma=0.95 finding. |
| 6 | gamma_low_mid | 1e-4 | 0.92 | 32 | 1.0 | 0.05 | 0.1 | -0.8 ± 4.45 | Baseline-level — further evidence of gamma insensitivity in the 0.9+ range. |
| 7 | batch_high | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | -2.4 ± 6.59 | Near baseline, consistent with larger batches being safe. |
| 8 | batch_mid | 1e-4 | 0.99 | 48 | 1.0 | 0.05 | 0.1 | +4.8 ± 4.17 | Positive result — mid-large batches performed well across all three members' sets. |
| 9 | batch_low | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.1 | -7.2 ± 3.43 | Mildly degraded — far less severe than the -51.2 the identical config produced in Jok's session. Small batches are high-variance: sometimes mildly worse, sometimes catastrophic. |
| 10 | eps_start_mid | 1e-4 | 0.99 | 32 | 0.8 | 0.05 | 0.1 | -17.8 ± 4.96 | Reduced initial exploration hurt, matching Wagner's eps_start finding. |
| 11 | eps_end_low | 1e-4 | 0.99 | 32 | 1.0 | 0.03 | 0.1 | -21.4 ± 6.25 | Less residual exploration degraded performance in this run. |
| 12 | eps_slow_decay | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.2 | -6.2 ± 2.93 | Slower decay slightly below baseline, consistent with Jok's finding at fraction=0.5. |
| 13 | eps_fraction_mid | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.15 | -13.8 ± 14.43 | Below baseline with very high variance. |
| 14 | gamma_batch_tuned | 1e-4 | 0.95 | 64 | 1.0 | 0.05 | 0.1 | +2.0 ± 6.90 | Combined tuning run: positive score, consistent with the group's tuned-combo pattern. |

## Discussion of Tuning Results

**Learning rate is the most sensitive hyperparameter, with a sharp cliff.** Across five values tested by the group: 5e-6 (-74.6) and 1e-5 (-6.4) learn too slowly; 1e-4 and 2e-4 perform well; 5e-4 (-25.0) already degrades; 1e-3 (-33.0) diverges. The safe zone is narrow — roughly 1e-4 to 2e-4 — and the penalty for exceeding it is severe because oversized gradient steps make Q-value estimates overshoot and never recover.

**Batch size showed the clearest group-wide pattern.** Large batches (48, 64, 128) were consistently safe or beneficial across all three members — smoother training curves from lower gradient noise. Small batches (8, 16) were consistently harmful, ranging from mildly worse (-7.2) to catastrophic (-51.2, the group's worst result). With few samples per update, each gradient is dominated by whichever random experiences were drawn, yanking the network in contradictory directions.

**Gamma matters far less than expected — because of Boxing's reward structure.** Values from 0.50 to 0.999 mostly performed near baseline. Boxing delivers a reward within a few frames of every punch, so an agent valuing only ~2 steps of future reward (gamma=0.50) can still play competently. Two runs (gamma=0.95 at +4.2, gamma=0.999 at +5.4) beat their baselines, but the same configs scored worse in other members' sessions — we read gamma as broadly insensitive here, in contrast to sparse-reward games where it is critical.

**Exploration changes mostly hurt.** Persistent high epsilon (0.2 forever) polluted the replay buffer with random-action transitions (-4.4); very low exploration (eps_end 0.01–0.03, eps_start 0.5–0.8) also degraded results; stretching decay over half of training delayed learning without improving it. The default schedule (1.0 → 0.05 over 10% of training) was near-optimal: Boxing's dense rewards make good behaviour easy to discover, so extended exploration wastes training budget.

**Run-to-run variance is real and large.** The same config (batch=16) scored -51.2 in one member's session and -7.2 in another's; gamma=0.999 scored +5.4 and -16.6. Fixed seeds do not guarantee identical results across GPU sessions because CUDA operations are non-deterministic and DQN compounds small divergences. This is why each member's observations are anchored to their own baseline, and why we favour patterns confirmed across multiple members over single-run results.

**Final configuration and why it wins:** `lr=1e-4` (centre of the narrow safe zone), `batch=64` (stability confirmed by three members independently), `gamma=0.99` (standard; gamma shown to be insensitive), default epsilon schedule (all deviations tested made things worse). At 100k steps this scored +7.4 — the best of all 35 group runs — and its curve was still climbing, so we trained it for 500,000 steps to produce the final model: **+30.00 ± 16.44**, an agent that decisively wins matches.

## Gameplay Video

Watch the agent playing via play.py (final model, 3 episodes: +27, +34, +33):
**[Gameplay video (Google Drive)](https://drive.google.com/drive/folders/1Y2cgqzKOOF8HYirpSn-sJJrRPAUHjA27)**

The recording shows play.py being launched in the terminal and the agent
(white boxer) pursuing and out-punching the opponent across full episodes.

## Evaluation (play.py)

play.py loads `models/dqn_model.zip`, recreates the environment with identical preprocessing to training (this is essential — the network expects 4 stacked 84x84 grayscale frames), and plays episodes with `render_mode="human"` for GUI display. Action selection uses `model.predict(obs, deterministic=True)`, which implements the greedy Q-policy: epsilon is ignored and the agent always takes the action with the highest Q-value.