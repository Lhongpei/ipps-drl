# Inference

## `InferenceEngine`

A single class that wraps a trained PPO checkpoint and exposes three search methods through one `solve()` call.

```python
from ipps_drl import InferenceEngine

engine = InferenceEngine(
    checkpoint="checkpoints/0605.pt",
    device="auto",          # also accepts "cuda:0" / "cpu"
    config=None,            # None = repo-root config.yaml
)

result = engine.solve("problem.ipps", method="greedy")
```

### Constructor

| arg | type | default | meaning |
| --- | --- | --- | --- |
| `checkpoint` | `str` / path | required | path to a `.pt` file produced by `scripts/train_drl.py` |
| `device` | `str` | `"auto"` | `"auto"` picks `cuda:0` when available, else `cpu` |
| `config` | path / `DictConfig` / `None` | `None` | only the `env_paras` / `nn_paras` sections are consulted; `None` loads the repo-root `config.yaml` |

### `engine.solve(problem, method, **kwargs)`

| method | extra kwargs (defaults) | description |
| --- | --- | --- |
| `"greedy"` | `return_schedule=True` | Deterministic argmax rollout. One trajectory. |
| `"sampling"` | `num_sample=25, num_average=1, return_schedule=True` | Build a batch of `num_sample` copies, sample stochastically, keep best. Repeat `num_average` times and keep the global best. |
| `"mcts"` | `time_limit=60, iteration_limit=None, exploration=5.0, lower_bound=-1, return_schedule=True` | Search with the policy as a prior. Time-bounded by default; supply `iteration_limit` instead for a fixed-budget search. |

Returns an [`InferenceResult`](#inferenceresult).

### `engine.solve_many(problems, method, **kwargs)`

Convenience loop:

```python
results = engine.solve_many(
    ["problem01.ipps", "problem02.ipps", "problem03.ipps"],
    method="sampling", num_sample=16,
)
```

## `InferenceResult`

```python
@dataclass
class InferenceResult:
    makespan: float
    schedule: np.ndarray | None      # (n_ops, 5) — [ope, ma, job, start, end]
    method: str                       # "greedy" / "sampling" / "mcts"
    wall_time_s: float
    problem: str | None
    extras: dict                      # method-specific extras
```

`extras` contents per method:

- `greedy`, `sampling`: `per_run_makespans`, `batch_size`
- `mcts`: `action_list`, `best_makespan_trace`, `round_makespan_trace`

## CLI

The shipped `scripts/test.py` is a thin wrapper over `InferenceEngine` for batch evaluation:

```bash
python scripts/test.py --method greedy
python scripts/test.py --method sampling --num_sample 25 --num_average 2
python scripts/test.py --checkpoint checkpoints/0605.pt --num_ins 3 --save_solutions
```

It iterates every `.pt` under `--checkpoint_dir` × every problem under `--data_dir`, writes one CSV per checkpoint to `--save_dir`. See `python scripts/test.py --help` for the full flag list.

For MCTS batch evaluation, use `scripts/run_mcts_batch.py` (same flags + `--time_limit`, `--exploration`, `--use_kim_lb`).
