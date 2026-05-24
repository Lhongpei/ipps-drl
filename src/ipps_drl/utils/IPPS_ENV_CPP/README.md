# IPPS_ENV_CPP

C++ implementation of the IPPS environment plus a Cython binding that exposes it
to Python (`env_wrapper`). Used by the MCTS inference path
(`ipps_drl.inference.mcts`).

The C++ core is a vendored copy of <https://github.com/Lhongpei/IPPS_ENV_CPP>.
The `pywrap/` directory adds the Cython binding and is specific to this repo.

## Layout

```
IPPS_ENV_CPP/
├── include/        # C++ headers
├── src/            # C++ sources
├── main.cpp        # Stand-alone C++ driver (optional, for debugging the env)
├── env_test.cpp    # ditto
├── test.cpp        # ditto
└── pywrap/
    ├── env_wrapper.pyx        # Cython source — public Python API
    ├── env_wrapper.pxd        # Cython declarations
    ├── setup.py               # build script (run from inside pywrap/)
    └── env_wrapper*.so        # built extension (generated)
```

## Build

The extension is required only for the MCTS inference path. Greedy and sampling
inference do not need it.

Prerequisites: a C++17 compiler (gcc 9+ or clang 10+), Python headers, and
Cython:

```bash
pip install cython
```

Build the extension in place:

```bash
cd src/ipps_drl/utils/IPPS_ENV_CPP/pywrap
python setup.py build_ext --inplace
```

This produces `env_wrapper.cpython-*.so` next to the source. The package import
path is `ipps_drl.utils.IPPS_ENV_CPP.pywrap.env_wrapper`.

To rebuild after editing the C++ sources or `env_wrapper.pyx`:

```bash
rm -rf build env_wrapper.cpp env_wrapper*.so
python setup.py build_ext --inplace
```

## Public Python API (env_wrapper)

```python
import ipps_drl.utils.IPPS_ENV_CPP.pywrap.env_wrapper as env_wrap

lines  = env_wrap.read_lines("data/test/kim/problem/problem01.ipps")
env    = env_wrap.PyEnv(lines, is_eval=True)

env.step(ope_id, mas_id)                # single step (ope_id=-1 means wait)
env.steps([(ope, mas), (ope, mas), ...]) # batched apply
env.check_done(); env.is_done()
env.get_cur_makespan(); env.get_time()
copy_env = env.copy()

# Greedy dispatch rule helper used by mcts.py rollouts
actions, makespan = env_wrap.run_greedy(env, ope_rule_type=1, ma_rule_type=1)
```

## Relationship with the upstream repo

The C++ headers / source in this directory are mostly identical to the upstream
[IPPS_ENV_CPP](https://github.com/Lhongpei/IPPS_ENV_CPP) repo, with two
deliberate local additions that the Cython binding relies on:

* `Env` has a copy constructor and assignment operator (so `PyEnv.copy()` can
  clone an env mid-search).
* `Env::printDebugInfo()` for ad-hoc inspection from Python.

The `steps()` batched-apply method matches upstream verbatim and is required by
the MCTS rollout (`ipps_drl.inference.mcts.evaluate`).

If you later sync newer upstream changes, keep the two additions above and the
`pywrap/` directory in place.
