"""IPPS-DRL: GNN-based deep reinforcement learning for Integrated Process Planning and Scheduling.

Quickest path for users::

    import ipps_drl

    # Environment
    env = ipps_drl.IPPSEnv(case=..., env_paras=...)

    # Inference
    engine = ipps_drl.InferenceEngine(checkpoint="checkpoints/0605.pt")
    result = engine.solve("problem.ipps", method="greedy")

    # Instance generation
    gen = ipps_drl.InstanceGenerator(ipps_drl.InstanceConfig(num_jobs=6, num_machines=5))
    instance = gen.from_pool("data/jobs/job_with_mas_5/")
"""

__version__ = "0.1.0"

from .env import IPPSEnv, EnvState, load_ipps, nums_detec
from .generator import (
    CaseGenerator,
    Instance,
    InstanceConfig,
    InstanceGenerator,
    Job,
    JobConfig,
    JobGenerator,
)
from .inference import InferenceEngine, InferenceResult

__all__ = [
    "__version__",
    # Environment
    "IPPSEnv",
    "EnvState",
    "load_ipps",
    "nums_detec",
    # Inference
    "InferenceEngine",
    "InferenceResult",
    # Generation
    "JobConfig",
    "Job",
    "JobGenerator",
    "InstanceConfig",
    "Instance",
    "InstanceGenerator",
    "CaseGenerator",
]
