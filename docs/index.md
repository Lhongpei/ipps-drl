# IPPS-DRL

GNN-based deep reinforcement learning for the **Integrated Process Planning and Scheduling** (IPPS) problem.

Official implementation of the paper *[Solving Integrated Process Planning and Scheduling Problem via Graph Neural Network Based Deep Reinforcement Learning](https://arxiv.org/abs/2409.00968)*.

## What's in the box

- A **GPU-parallel IPPS environment** that returns heterogeneous-graph states.
- A **PPO trainer** (`scripts/train_drl.py`) and a **behaviour-cloning trainer** (`scripts/train_bc.py`).
- A unified **`InferenceEngine`** that exposes three inference methods through a single `solve()` call:
  - `greedy` — deterministic argmax rollout (DRL-G)
  - `sampling` — parallel sample rollouts, keep best (DRL-S)
  - `mcts` — search guided by the trained policy, simulated by a C++ env
- A clean **random-instance generator** (`JobGenerator`, `InstanceGenerator`).
- Two external **MILP baselines** (OR-Tools, Gurobi) and a set of hand-crafted greedy dispatching rules.

## 30-second tour

```python
import ipps_drl

# 1. Load a trained checkpoint and run greedy inference on one instance.
engine = ipps_drl.InferenceEngine(checkpoint="checkpoints/0605.pt")
result = engine.solve("data/test/kim/problem/problem01.ipps", method="greedy")
print(result.makespan, result.schedule.shape)

# 2. Generate a fresh random IPPS instance.
gen = ipps_drl.InstanceGenerator(ipps_drl.InstanceConfig(num_jobs=6, num_machines=5))
instance = gen.from_pool("data/jobs/job_with_mas_5/")
instance.save("my_problem.ipps")

# 3. Step the environment yourself.
env = ipps_drl.IPPSEnv(case=[instance.lines], env_paras=..., data_source="tensor")
state = env.reset()
state, reward, done, _ = env.step(actions)
```

Continue to the [Quickstart](quickstart.md) for the full install/run path, or jump to:

- [**Inference**](inference.md) — `InferenceEngine` API + the three search methods.
- [**Training**](training.md) — PPO and behaviour cloning end-to-end.
- [**Generators**](generators.md) — produce jobs and instances on demand.
- [**Architecture**](architecture.md) — what each subpackage does and how they fit together.
- [**File format**](file-format.md) — the `.ipps` / `.ippssol` format spec.
