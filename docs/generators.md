# Generators

Two layers: a **single job** (one DAG of operations) and a full **IPPS instance** (several jobs combined, each with its own machine assignments).

## Job generation

```python
from ipps_drl import JobGenerator, JobConfig

cfg = JobConfig(
    machine_range=(5, 5),      # min, max machines available
    mas_p=0.5,                  # P(a given machine can process a given op)
    randomize_opes=False,       # if True, draw ope_num from `ope_range`
    ope_num=10,                 # base operation count
    or_num=2,                   # number of OR splits to inject
    and_p=0.3,                  # P(an OR branch gets nested AND branches)
    time_range=(100, 500),      # per-op mean processing time
)
job = JobGenerator(cfg).generate()
print(job.num_machines, job.num_opes)
print(job.lines[:4])

# Save with the canonical "_mas_<N>.txt" suffix the pool loader expects:
job.save("data/jobs/job_with_mas_5/job_42")  # → "job_42_mas_5.txt"
```

### `JobConfig` knobs

| group | fields |
| --- | --- |
| Machines | `machine_range`, `mas_p`, `time_range`, `time_bias` |
| DAG skeleton | `max_out`, `alpha`, `beta` |
| Operation count | `randomize_opes`, `ope_num`, `ope_range`, `total_ope_range` |
| OR splits | `or_num`, `or_p`, `or_road_num`, `ope_num_orpath` |
| AND inside OR | `and_p`, `and_road_num`, `ope_num_andpath` |

## Instance generation

```python
from ipps_drl import InstanceGenerator, InstanceConfig

gen = InstanceGenerator(InstanceConfig(num_jobs=6, num_machines=5))

# Sample 6 jobs from a pre-generated pool on disk
inst = gen.from_pool("data/jobs/job_with_mas_5/")

# Or roll fresh jobs and combine on the fly
inst = gen.from_fresh_jobs(JobConfig(or_num=2))

inst.save("my_problem.ipps")
print(inst.num_jobs, inst.num_machines, inst.num_opes)
print(inst.source_job_ids)        # populated by from_pool
```

`from_pool` re-rolls until the selected jobs collectively use exactly `num_machines` machines. The cap is controlled by `InstanceConfig.max_resample_attempts` (`None` = unbounded).

## Legacy `CaseGenerator`

For backwards compatibility with `scripts/train_drl.py` and any of your own old scripts:

```python
from ipps_drl import CaseGenerator
case = CaseGenerator(job_num=5, machine_num=3, job_folder="data/jobs/job_with_mas_3/")
lines = case.get_case()           # list[str], same format as before
```

Internally this just wraps `InstanceGenerator`.

## Bundled CLIs

```bash
# Roll a fresh job pool (4500 jobs at 5 machines each)
python -m ipps_drl.generator.jobs_generator \
    --out_dir data/jobs/job_with_mas_5 --count 4500 --machines 5 --max_or 3

# Assemble 100 IPPS instances from a pool
python -m ipps_drl.generator.case_generator_ipps \
    --job_folder data/jobs/job_with_mas_5 --num_jobs 6 --num_machines 5 --count 100
```

Both honour `--seed`.
