"""High-level inference API for IPPS-DRL.

Three methods share one ``InferenceEngine``:

* ``greedy``   — DRL-G: roll the trained policy forward with argmax.
* ``sampling`` — DRL-S: roll out ``num_sample`` parallel copies of the same instance,
                 sample stochastically, keep the best.
* ``mcts``     — Search with the learned policy as a prior. Requires the optional
                 C++ env wrapper (see ``src/ipps_drl/utils/IPPS_ENV_CPP``).

Typical use::

    from ipps_drl.inference import InferenceEngine

    engine = InferenceEngine(checkpoint="checkpoints/0605.pt", device="cuda:0")

    result = engine.solve("data/test/kim/problem/problem1.txt", method="greedy")
    print(result.makespan, result.schedule.shape)

    # Parallel sampling
    result = engine.solve("problem.ipps", method="sampling", num_sample=25)

    # MCTS with time budget
    result = engine.solve("problem.ipps", method="mcts", time_limit=60, exploration=5)

    # Batched
    results = engine.solve_many([f"p{i}.ipps" for i in range(10)], method="greedy")
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from ipps_drl.env.ipps_env import IPPSEnv
from ipps_drl.env.load_data import nums_detec
from ipps_drl.models.memory import MemoryRL
from ipps_drl.models.ppo import PPO

from .result import InferenceResult


PathLike = Union[str, os.PathLike]


_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config.yaml"


class InferenceEngine:
    """Wraps a trained PPO model and exposes one ``solve()`` method per instance.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint produced by :mod:`scripts.train_drl`.
        device: Torch device string. ``"auto"`` (default) picks ``cuda:0`` if available.
        config: Either a path to a ``config.yaml``-style file, an already-loaded
            :class:`omegaconf.DictConfig`, or ``None`` to use the repo-root default.
            Only the ``env_paras`` / ``nn_paras`` sections are consulted.
    """

    def __init__(
        self,
        checkpoint: PathLike,
        device: str = "auto",
        config: Union[PathLike, DictConfig, None] = None,
    ) -> None:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        cfg = self._load_config(config)
        self.env_paras = copy.deepcopy(cfg.env_paras)
        self.env_paras["device"] = self.device
        self.model_paras = copy.deepcopy(cfg.nn_paras)
        self.model_paras["device"] = self.device
        self.train_paras = cfg.train_paras  # PPO __init__ reads this

        # Build the model inside a device context so all parameter creations land
        # on the right device without relying on torch.set_default_device.
        with torch.device(self.device):
            self.model = PPO(self.model_paras, self.train_paras, num_envs=1)
        state = torch.load(str(checkpoint), map_location=self.device)
        self.model.policy.load_state_dict(state)
        self.model.policy_old.load_state_dict(state)
        self.model.policy.eval()
        self.model.policy_old.eval()
        self.checkpoint = str(checkpoint)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _load_config(config) -> DictConfig:
        if isinstance(config, DictConfig):
            return config
        path = Path(config) if config is not None else _DEFAULT_CONFIG
        return OmegaConf.load(str(path))

    @staticmethod
    def _read_problem_lines(problem: PathLike) -> List[str]:
        with open(problem) as f:
            return f.read().splitlines()

    def _build_env(self, problem: PathLike, *, batch_size: int, copy_one: bool) -> IPPSEnv:
        """Create a fresh ``IPPSEnv`` for a single problem file.

        ``copy_one=True`` packs ``batch_size`` copies of the same instance into the
        batch (used by sampling); otherwise the batch contains a single instance.
        """
        lines = self._read_problem_lines(problem)
        num_jobs, num_mas, _ = nums_detec(lines)
        env_paras = copy.deepcopy(self.env_paras)
        env_paras["batch_size"] = batch_size
        env_paras["num_jobs"] = num_jobs
        env_paras["num_mas"] = num_mas
        if copy_one:
            return IPPSEnv(case=str(problem), env_paras=env_paras, data_source="copy_file")
        return IPPSEnv(case=[str(problem)], env_paras=env_paras, data_source="file")

    # ------------------------------------------------------------------ public

    def solve(
        self,
        problem: PathLike,
        method: str = "greedy",
        **method_kwargs,
    ) -> InferenceResult:
        """Solve a single problem with the requested method.

        Args:
            problem: Path to a ``.ipps`` instance file.
            method: One of ``"greedy"``, ``"sampling"``, ``"mcts"``.
            **method_kwargs: Forwarded to the per-method helper. Recognised keys:

                * ``return_schedule`` (bool, default ``True``): include the full
                  schedule in the result. Set ``False`` to skip the (mildly expensive)
                  ``env.get_schedule`` extraction when only ``makespan`` is needed.
                * For ``sampling``: ``num_sample`` (default 25), ``num_average``
                  (default 1, number of independent sample batches; keeps the
                  overall best across all batches).
                * For ``mcts``: ``time_limit`` seconds (default 60),
                  ``iteration_limit`` (alternative budget), ``exploration``
                  (default 5), ``lower_bound`` (default ``-1``, no LB).
        """
        method = method.lower()
        if method == "greedy":
            return self._solve_drl(problem, sample=False, **method_kwargs)
        if method == "sampling":
            return self._solve_drl(problem, sample=True, **method_kwargs)
        if method == "mcts":
            return self._solve_mcts(problem, **method_kwargs)
        raise ValueError(f"Unknown method {method!r}. Use 'greedy', 'sampling' or 'mcts'.")

    def solve_many(
        self,
        problems: Iterable[PathLike],
        method: str = "greedy",
        **method_kwargs,
    ) -> List[InferenceResult]:
        """Sequentially solve a list of problems with the same method/settings."""
        return [self.solve(p, method=method, **method_kwargs) for p in problems]

    # ----------------------------------------------------------- DRL roll-out

    def _solve_drl(
        self,
        problem: PathLike,
        *,
        sample: bool,
        num_sample: int = 25,
        num_average: int = 1,
        return_schedule: bool = True,
    ) -> InferenceResult:
        batch_size = num_sample if sample else 1
        env = self._build_env(problem, batch_size=batch_size, copy_one=sample)
        memory = MemoryRL()

        best_makespan = float("inf")
        best_schedule = None
        per_run_makespans: List[float] = []

        wall_start = time.time()
        for _ in range(max(1, num_average)):
            env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    actions = self.model.policy_old.act(
                        env.state, memory, flag_sample=sample, flag_train=False
                    )
                _, _, dones, _ = env.step(actions)
                done = bool(dones.all().item())
            run_best_idx = int(env.makespan_batch.argmin().item())
            run_best_ms = float(env.makespan_batch[run_best_idx].item())
            per_run_makespans.append(run_best_ms)
            if run_best_ms < best_makespan:
                best_makespan = run_best_ms
                best_schedule = env.get_schedule(run_best_idx) if return_schedule else None
        wall = time.time() - wall_start

        # Sanity-check: schedule should validate.
        ok, _ = env.validate_gantt()
        if not ok:
            raise RuntimeError(f"validate_gantt failed for {problem}")

        method_label = "sampling" if sample else "greedy"
        return InferenceResult(
            makespan=best_makespan,
            schedule=best_schedule,
            method=method_label,
            wall_time_s=wall,
            problem=str(problem),
            extras={"per_run_makespans": per_run_makespans, "batch_size": batch_size},
        )

    # ----------------------------------------------------------------- MCTS

    def _solve_mcts(
        self,
        problem: PathLike,
        *,
        time_limit: float = 60.0,
        iteration_limit: Optional[int] = None,
        exploration: float = 5.0,
        lower_bound: float = -1,
        return_schedule: bool = True,
        strong: bool = False,
        selection_mode: str | None = None,
        puct_mode: str | None = None,
        dirichlet_alpha: float | None = None,
        dirichlet_eps: float | None = None,
        cache_rollouts: bool | None = None,
        parallel_rollouts: bool | None = None,
    ) -> InferenceResult:
        # MCTS needs the optional C++ wrapper. Import lazily so the rest of the
        # engine still works on machines that haven't built the extension.
        try:
            from ipps_drl.inference.mcts import mcts as MCTS
            from ipps_drl.utils.IPPS_ENV_CPP.pywrap import env_wrapper as env_wrap
        except ImportError as exc:
            raise RuntimeError(
                "MCTS inference requires the optional C++ env wrapper. "
                "Build it under src/ipps_drl/utils/IPPS_ENV_CPP/pywrap "
                "(see https://github.com/Lhongpei/IPPS_ENV_CPP for source)."
            ) from exc

        env = self._build_env(problem, batch_size=1, copy_one=False)
        env.reset()

        # Parse the per-job start operation IDs from the .ipps file's `info` block.
        lines = env_wrap.read_lines(str(problem))
        start_nodes: List[int] = []
        for line in lines:
            if line.endswith("start"):
                parts = line.split()
                if parts and parts[0].isdigit():
                    start_nodes.append(int(parts[0]))

        cy_env = env_wrap.PyEnv(lines, is_eval=True)

        # mcts.search uses `time.time() + self.timeLimit` — pass seconds directly.
        agent = MCTS(
            self.model,
            timeLimit=time_limit if iteration_limit is None else None,
            iterationLimit=iteration_limit,
            explorationConstant=exploration,
            strong=strong,
            selection_mode=selection_mode,
            puct_mode=puct_mode,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            cache_rollouts=cache_rollouts,
            parallel_rollouts=parallel_rollouts,
        )

        wall_start = time.time()
        action_list, best_makespan = agent.search(env, cy_env, start_nodes, lower_bound)
        wall = time.time() - wall_start

        schedule = env.get_schedule(0) if return_schedule else None
        return InferenceResult(
            makespan=float(best_makespan),
            schedule=schedule,
            method="mcts",
            wall_time_s=wall,
            problem=str(problem),
            extras={
                "action_list": action_list,
                "best_makespan_trace": getattr(agent, "bestmakespan_list", None),
                "round_makespan_trace": getattr(agent, "makespan_list", None),
            },
        )
