"""IPPS job and instance generators.

Quick usage::

    from ipps_drl.generator import (
        JobGenerator, JobConfig,
        InstanceGenerator, InstanceConfig,
    )

    # One random job (no I/O)
    job = JobGenerator(JobConfig(machine_range=(5, 5))).generate()
    job.save("data/jobs/job_with_mas_5/job_42")  # -> "job_42_mas_5.txt"

    # One IPPS instance assembled from a pool of jobs
    gen = InstanceGenerator(InstanceConfig(num_jobs=6, num_machines=5))
    inst = gen.from_pool("data/jobs/job_with_mas_5/")
    inst.save("out.ipps")
"""

from .case_generator_ipps import (
    CaseGenerator,
    Instance,
    InstanceConfig,
    InstanceGenerator,
)
from .jobs_generator import (
    Job,
    JobConfig,
    JobGenerator,
    jobs_generator,
)

__all__ = [
    "Job",
    "JobConfig",
    "JobGenerator",
    "jobs_generator",
    "Instance",
    "InstanceConfig",
    "InstanceGenerator",
    "CaseGenerator",
]
