# Architecture

## Repository layout

```
ipps-drl/
├── src/ipps_drl/          # installable package (pip install -e .)
│   ├── env/               # IPPSEnv, .ipps loader
│   ├── network/           # Heterogeneous GAT + Graph_Batch
│   ├── models/            # PPO, BC, policy, replay memory
│   ├── inference/         # InferenceEngine + MCTS
│   ├── generator/         # JobGenerator + InstanceGenerator
│   ├── greedy/            # Hand-crafted dispatching rules
│   ├── utils/             # Tensor padding helpers, gantt drawing, C++ env wrapper
│   ├── validate.py        # validate() / get_validate_env() used by trainers
│   └── dataset.py         # Dataset used by behaviour cloning
├── scripts/               # CLI entry points
├── baselines/             # OR-Tools / Gurobi MILP solvers
├── data/                  # Problem / solution data + job pools
├── checkpoints/           # Pre-trained .pt files
└── config.yaml
```

## Module map

### `ipps_drl.env`

```python
from ipps_drl import IPPSEnv, EnvState, load_ipps, nums_detec
```

`IPPSEnv` is a vectorised batched environment. The user-facing surface is exactly:

- `step(actions)` — apply the chosen `(operation, machine, job)` per batch row; `-1` is a wait action.
- `reset()` — restore to the initial state captured at `__init__` time.
- `validate_gantt()` — check that the produced schedule is feasible.
- `get_schedule(batch_idx)` — extract the schedule for one batch item as `[op, ma, job, start, end]` rows.
- `find_eligible_pairs(find_future=False)` — `(B, num_opes, num_mas)` bool tensor of eligible `(op, ma)` pairs.
- `add_job(...)` — add a new job mid-episode (dynamic-scheduling experiments).
- `proc_time_change(...)` — perturb processing times mid-episode.

Everything else on `IPPSEnv` is private (`_`-prefixed).

### `ipps_drl.network`

```python
from ipps_drl import Graph_Batch, GraphEmbedding, Actor, Critic
```

- `Graph_Batch` — PyG-backed batched heterogeneous graph with in-place feature / edge-subgraph updates.
- `GraphEmbedding` — stack of `HeteroConv(GATv2Conv)` layers + `JumpingKnowledge`.
- `Actor` / `Critic` — small MLPs.

### `ipps_drl.models`

```python
from ipps_drl import PPO  # plus: BehaviorCloning, MemoryRL, MemoryIL, Policy, DRLPolicy, ExpertPolicy, Expert
```

- `Policy` carries the shared embedding / probability machinery.
- `DRLPolicy` adds `act()` (sample) and `action_with_prob()` (priors for MCTS).
- `ExpertPolicy` is used during imitation data generation.
- `PPO` runs PPO updates on `MemoryRL`-collected trajectories; `BehaviorCloning` does the IL equivalent on `MemoryIL`.

### `ipps_drl.inference`

```python
from ipps_drl import InferenceEngine, InferenceResult
```

The user-facing inference entry-point — see [Inference](inference.md). MCTS lives at `ipps_drl.inference.mcts` and is imported lazily by the engine so that machines without the C++ extension can still use greedy / sampling.

### `ipps_drl.generator`

```python
from ipps_drl import JobGenerator, JobConfig, InstanceGenerator, InstanceConfig
```

See [Generators](generators.md).

### `ipps_drl.greedy`

```python
from ipps_drl.greedy import greedy_rule
```

Hand-crafted priority rules (SPT, MOR, MWKR, FIFO, EFT, LUM, …). Used as baselines and as the rollout heuristic during MCTS simulation (via the C++ `run_greedy` driver).

### `ipps_drl.utils.IPPS_ENV_CPP`

C++ implementation of the IPPS environment (vendored from
[Lhongpei/IPPS_ENV_CPP](https://github.com/Lhongpei/IPPS_ENV_CPP)) plus a Cython
binding (`env_wrapper`). Only required by MCTS inference. Build with:

```bash
pip install cython
cd src/ipps_drl/utils/IPPS_ENV_CPP/pywrap
python setup.py build_ext --inplace
```

## State flow (one PPO iteration)

```
CaseGenerator.get_case()          # sample N instances from data/jobs/...
        ↓
IPPSEnv(case=...).reset()
        ↓
loop until env.done_batch.all():
    actions = policy.act(state, memory)
    state, reward, done, _ = env.step(actions)
        ↓
PPO.update(memory)                # every `update_timestep` iterations
        ↓
validate(env_valid, policy)       # every `save_timestep` iterations
        ↓
torch.save(policy.state_dict())   # if new best
```

## What was optimised vs. the original release

A handful of changes brought env.step from ~10 ms to ~6 ms and the policy forward from ~37 ms to ~28 ms (Kim instances on a single A100), end-to-end ~1.4× per-step:

- Skip `update_edge_sub_graph` when neither `remain_opes` nor `combs_id` changed.
- Replace `torch.isin` with a bool-table lookup in the per-edge-type loop.
- Cache `find_eligible_pairs` across the inner `if_no_eligible` / `update_state` calls in `env.step`.
- Replace `torch.set_default_device` (per-op Python wrapper) with explicit `device=` factory calls.
- Cache singleton `GraphNorm` modules instead of constructing them inside the per-step `normalize_hetero_data`.
- Snapshot `Batch` in replay memory via a direct tensor `.clone()` walk instead of `copy.deepcopy(to_data_list())`.
- Vectorise the per-batch action selection in `DRLPolicy.act`.
