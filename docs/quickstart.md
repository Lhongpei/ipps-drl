# Quickstart

## Install

```bash
git clone https://github.com/Lhongpei/ipps-drl.git
cd ipps-drl

# A fresh env is recommended; CUDA wheels for torch + torch_geometric must
# match your local CUDA version (see the torch_geometric install page).
conda create -n ipps-drl python=3.10 -y && conda activate ipps-drl

pip install -e .
```

After install, the package is importable as `ipps_drl` and three CLI scripts live under `scripts/` and `baselines/`.

!!! tip "MCTS inference is optional"
    The greedy / sampling inference paths work out of the box. MCTS additionally
    needs the C++ environment wrapper at
    `src/ipps_drl/utils/IPPS_ENV_CPP/pywrap` — see
    [its README](https://github.com/Lhongpei/ipps-drl/blob/main/src/ipps_drl/utils/IPPS_ENV_CPP/README.md)
    for the one-time build command.

## Run inference on the bundled Kim benchmark

```bash
# Greedy
python scripts/test.py --method greedy

# Sampling: 25 parallel rollouts, take the best, repeat 2× and keep the global best
python scripts/test.py --method sampling --num_sample 25 --num_average 2

# Or use the Python API directly:
python - <<'PY'
import ipps_drl
engine = ipps_drl.InferenceEngine(checkpoint="checkpoints/0605.pt")
result = engine.solve("data/test/kim/problem/problem01.ipps", method="greedy")
print(result.makespan, result.schedule.shape)
PY
```

CSV summaries land under `save/test_<timestamp>/`.

## Train a fresh PPO model

```bash
python scripts/train_drl.py
```

All hyper-parameters live in [`config.yaml`](https://github.com/Lhongpei/ipps-drl/blob/main/config.yaml). Weights & Biases logging is opt-in via `use_wandb = True` at the top of the script.

Training writes checkpoints + per-iteration validation results into `save/train_<timestamp>/`.

## Generate a custom instance

```python
from ipps_drl import JobGenerator, JobConfig, InstanceGenerator, InstanceConfig

# Bulk-roll 100 jobs into a pool
gen = JobGenerator(JobConfig(machine_range=(5, 5), or_num=2))
for i in range(100):
    gen.generate().save(f"data/jobs/my_pool/job_{i}")

# Assemble a 6-job × 5-machine IPPS instance from that pool
inst = InstanceGenerator(InstanceConfig(num_jobs=6, num_machines=5)).from_pool("data/jobs/my_pool")
inst.save("my_problem.ipps")
print(inst.num_opes, inst.source_job_ids)
```

Or use the bundled CLI:

```bash
python -m ipps_drl.generator.jobs_generator --out_dir data/jobs/my_pool --count 100 --machines 5
python -m ipps_drl.generator.case_generator_ipps --job_folder data/jobs/my_pool --count 50
```
