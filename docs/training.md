# Training

Two trainers, both configured by [`config.yaml`](https://github.com/Lhongpei/ipps-drl/blob/main/config.yaml).

## PPO (`scripts/train_drl.py`)

```bash
python scripts/train_drl.py
```

What it does:

1. Builds a fresh `PPO` model from the `nn_paras` section of `config.yaml`.
2. Every `parallel_iter` iterations, samples a fresh batch of training instances via `CaseGenerator` from `data/jobs/job_with_mas_<N>/`.
3. Each iteration: rolls the policy through an `IPPSEnv` episode, stores the trajectory into `MemoryRL`.
4. Every `update_timestep` iterations: runs `PPO.update()` on the accumulated memory.
5. Every `save_timestep` iterations: runs validation on `data/dev/<jobs><mas>/` and snapshots the best policy.

Outputs go to `save/train_<timestamp>/`:

- `save_best_<jobs>_<mas>_<iter>.pt` — newest best checkpoint
- `training_ave_<timestamp>.xlsx` — validation makespan curve
- `training_100_<timestamp>.xlsx` — per-validation-instance breakdown

[Weights & Biases](https://wandb.ai/) logging is gated by the `use_wandb` flag at the top of the script.

## Behaviour cloning (`scripts/train_bc.py`)

```bash
python scripts/train_bc.py
```

Imitation training from `(problem, solution)` pairs under `IL_test/<jobs><mas>/`. Uses `ipps_drl.dataset.ILDataScheduler` to assemble batches and `ipps_drl.models.BehaviorCloning` to fit the policy.

## Key config sections

```yaml
env_paras:
  num_jobs: 5             # training instance size
  num_mas: 3
  batch_size: 3           # parallel training instances
  is_greedy: false        # policy mode during rollout
  reward_info:            # which proxy reward to use
    estimate: false       # false = real makespan; true = estimated
    ma_mean: false        # min vs mean over candidate machines
    comb_mean: false      # min vs mean over OR-combinations

nn_paras:
  graph_embedding:
    model: GAT
    num_heads: [2, 2, 2]
    hidden_dim: 64

train_paras:
  lr: 5e-4
  K_epochs: 3
  eps_clip: 0.3
  max_iterations: 30000
  update_timestep: 5
  save_timestep: 10
  parallel_iter: 20
```

See the full file in the repo root.

## Resuming from a checkpoint

The training scripts don't auto-resume by default; uncomment the few lines near the top of `scripts/train_drl.py` that read:

```python
# pretrain_model_path = os.path.join('./model', os.listdir('./model')[0])
# model.policy.load_state_dict(torch.load(pretrain_model_path))
# model.policy_old.load_state_dict(torch.load(pretrain_model_path))
```

and point them at your checkpoint.

## Validating without training

```python
from ipps_drl.validate import get_validate_env, validate
from omegaconf import OmegaConf

conf = OmegaConf.load("config.yaml")
env_paras = conf.env_paras; env_paras["device"] = "cuda:0"
env = get_validate_env(env_paras)
mean_makespan, per_inst = validate(env_paras, env, policy, draw=False, gantt_path=".", N=0)
```
