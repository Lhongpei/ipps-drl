# IPPS-DRL

## Introduction

In this repository, we implement a DRL environment used for Integrated processing plans and scheduling problems. Our environment supports GPU parallel and returns a graph type as states used in both the DRL algorithm and greedy dispatching rules. Also, we implement functions to simulate the dynamic real world including adding new jobs and varying processing time during scheduling. All baselines and suggested methods can be found in this repository.

### DRL and IL implementation

Under `./models`, we implement the PPO algorithm, Behavior Cloning algorithm and policy classes.

`./models/policy.py` implements policy classes:

- `Policy` contains basic methods such as transforming embeddings of O-M pairs from graph to tensor, pooling embeddings and catting embedding and obtain action probabilities.
  
- `DRLPolicy` and `ExpertPolicy` inherit class `Policy`, which implements methods used in DRL and IL respectively.

`./models/Expert.py` contains the class `Expert` warping `ExpertPolicy`, which is used to follow expert actions.

`./models/memory.py` contains 2 classes, `MemoryRL` and `MemoryIL` recording trajectories used for DRL and IL respectively.

`./models/ppo.py` and `./models/bc.py` implement PPO and Behavior Cloning algorithms.

### Neural Network and State

Neural networks and the class of heterogeneous graphs representing states are implemented under `./network/`.

`./network/hetero_data.py` implements a heterogeneous graph class supporting batch processing updating to accelerate.

`./network/models.py` implements the Heterogeneous GAT, the MLP-based Actor and Critic.

The classes above are implemented based on [pytorch_geometric]([PyG](https://github.com/pyg-team/pytorch_geometric)), an open-source package for graph neural networks

### Environment

The environment and other utils for data are implemented under `./env/` including the data loader, data generator and the environment.

`./env/ipps_env.py` contains `IPPSEnv` a class of the environment containing the following function:

- `self.__init__`: Initialize the environment by the received problem that can be fjsp or ipps.

- `self.step`: Update the environment according to the action.

- `self.backup4reset`: Record the current status to allow resetting the environment to this status.

- `self.reset`: Reset the environment to the initial (or recorded) status.

- `self.add_job`: Add a job into the environment.

- `self.proc_time_change`: Change the processing time used for machines to process operations.

- `self.validate_gantt`: Validate whether a schedule is legal.

- `self.get_schedule`: Output the scheduling in the format of solutions.

### Problem Generator

Our problem generation contains 2 parts:

- Job Generation: We implement it in `./generator/Jobs_Generator.py`, which first generates a Direct acyclic graph, and then generates processing time randomly. We use [DAG_Gnerator](https://github.com/Livioni/DAG_Generator), an open-source package that supports randomly generating DAGs, in Jobs_Generator
  
- Problem Generation: It can be found in both `./generator/case_generator_ipps.py, which converts several job files into problem files.

### Greedy Dispatching rules

greedy dispatching rules are implemented in `./greedy/greedy_rules`.

Use the following command to run greedy solving test problems:

```python
python greedy_test.py
```

### MILP Models

Our MILP model is implemented based on [OR-Tools](https://github.com/google/or-tools), an open-source solver.

Use the following command to run MILP solver:

```python
python ipps_solve_ortools.py
```

## Reference

https://github.com/pyg-team/pytorch_geometric

https://github.com/songwenas12/fjsp-drl

https://github.com/Livioni/DAG_Generator

https://github.com/google/or-tools