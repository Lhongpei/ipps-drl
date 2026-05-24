"""Combine jobs into a full IPPS instance.

Public API:
    - :class:`InstanceConfig` — what kind of instance to assemble.
    - :class:`Instance`       — generated instance (lines + ``save()``).
    - :class:`InstanceGenerator` — ``.from_pool(dir)`` or ``.from_fresh_jobs(...)``.
    - :class:`CaseGenerator`  — legacy shim used by ``scripts/train_drl.py``.

Example::

    from ipps_drl.generator import InstanceGenerator, InstanceConfig

    gen = InstanceGenerator(InstanceConfig(num_jobs=6, num_machines=5))
    inst = gen.from_pool("data/jobs/job_with_mas_5/")
    inst.save("out.ipps")
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .jobs_generator import JobConfig, JobGenerator


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass
class InstanceConfig:
    """Configuration for assembling an IPPS instance from a set of jobs.

    Attributes:
        num_jobs: How many jobs to combine.
        num_machines: How many machines the instance must use. Generation re-rolls
            until exactly this many machines appear across the chosen jobs.
        max_resample_attempts: Safety cap on the re-roll loop. ``None`` = unbounded.
    """

    num_jobs: int = 5
    num_machines: int = 15
    max_resample_attempts: int | None = None


@dataclass
class Instance:
    """An assembled IPPS instance, ready to write to a ``.ipps`` file."""

    lines: List[str]
    num_jobs: int
    num_machines: int
    num_opes: int
    source_job_ids: List[int] | None = None  # populated by ``from_pool``

    def to_text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_text())
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r'\b\d+\b')
_PAREN_NUM_RE = re.compile(r'\((\d+)\)')


def _offset_line(line: str, offset: int) -> str:
    """Add ``offset`` to every bare integer in ``line``.

    A single-int parenthesised group ``(N)`` is preserved without offset because
    those mark OR-connector ids that must stay job-local.
    """
    out = _NUM_RE.sub(lambda m: str(int(m.group()) + offset), line)
    out = _PAREN_NUM_RE.sub(lambda m: f"({int(m.group(1)) - offset})", out)
    return out


def _section(lines: Sequence[str], start_marker: str, end_marker: str | None) -> List[str]:
    """Return the slice between markers (exclusive). End is EOF when ``end_marker`` is None."""
    # Each ``lines`` may use either bare markers (in-memory jobs) or "...\n" markers
    # (file-read jobs). Normalise.
    norm = [ln.rstrip('\n') for ln in lines]
    try:
        s = norm.index(start_marker) + 1
    except ValueError as exc:
        raise ValueError(f"Job is missing marker {start_marker!r}") from exc
    if end_marker is None:
        return norm[s:]
    e = norm.index(end_marker, s)
    return norm[s:e]


def _read_job_lines(path: Path) -> List[str]:
    return path.read_text().splitlines()


# ---------------------------------------------------------------------------
# Combine logic (shared by both code paths)
# ---------------------------------------------------------------------------


def _combine_jobs(
    job_line_lists: Sequence[Sequence[str]],
    num_machines: int,
) -> Instance | None:
    """Assemble one IPPS instance from a list of jobs.

    Returns ``None`` if the union of machines used by the jobs is not exactly
    ``{1, ..., num_machines}``; the caller should re-roll the selection.
    """
    current_offset = 0
    combined_out: List[str] = []
    combined_in: List[str] = []
    combined_info: List[str] = []
    all_machines: set[int] = set()

    for lines in job_line_lists:
        # ``out`` section
        for ln in _section(lines, "out", "in"):
            if ln.strip():
                combined_out.append(_offset_line(ln, current_offset))
        # ``in`` section
        for ln in _section(lines, "in", "info"):
            if ln.strip():
                combined_in.append(_offset_line(ln, current_offset))
        # ``info`` section
        for ln in _section(lines, "info", None):
            if not ln.strip():
                continue
            parts = ln.split()
            parts[0] = str(int(parts[0]) + current_offset)
            combined_info.append(' '.join(parts))
            all_machines.update(int(m) for m in parts[2::2])
        current_offset += int(lines[0].split()[2])

    if all_machines != set(range(1, num_machines + 1)):
        return None

    header = f"{len(job_line_lists)} {len(all_machines)} {current_offset}"
    lines = [header, "out", *combined_out, "in", *combined_in, "info", *combined_info]
    return Instance(
        lines=lines,
        num_jobs=len(job_line_lists),
        num_machines=len(all_machines),
        num_opes=current_offset,
    )


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------


class InstanceGenerator:
    """Build IPPS instances from either an on-disk job pool or freshly-rolled jobs."""

    def __init__(self, config: InstanceConfig | None = None) -> None:
        self.config = config or InstanceConfig()

    # ---- pool-based ----
    def from_pool(self, job_folder: str | Path) -> Instance:
        """Sample ``config.num_jobs`` files from ``job_folder`` and combine them.

        Only files matching the pattern ``..._mas_<num_machines>.txt`` are eligible.
        The selection is re-rolled until the union of machines is exactly
        ``{1, ..., num_machines}``.
        """
        folder = Path(job_folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder} does not exist")

        suffix = f"mas_{self.config.num_machines}"
        candidates = sorted(p for p in folder.iterdir()
                            if p.is_file() and p.suffix == ".txt" and suffix in p.name)
        if len(candidates) < self.config.num_jobs:
            raise ValueError(
                f"{folder} has only {len(candidates)} files matching '*{suffix}*.txt'; "
                f"need at least {self.config.num_jobs}"
            )

        for attempt in self._attempt_loop():
            picked = sorted(random.sample(range(len(candidates)), self.config.num_jobs))
            jobs = [_read_job_lines(candidates[i]) for i in picked]
            instance = _combine_jobs(jobs, self.config.num_machines)
            if instance is None:
                continue
            instance.source_job_ids = [
                int(re.search(r'job_(\d+)_mas', candidates[i].name).group(1))
                for i in picked
            ]
            return instance
        raise RuntimeError("exceeded max_resample_attempts without a valid combination")

    # ---- fresh-jobs ----
    def from_fresh_jobs(self, job_config: JobConfig | None = None) -> Instance:
        """Generate ``config.num_jobs`` jobs on the fly and combine them.

        ``job_config.machine_range`` is overridden to ``(num_machines, num_machines)``
        so every fresh job uses the right number of machines.
        """
        base = job_config or JobConfig()
        base = JobConfig(
            **{**base.__dict__,
               "machine_range": (self.config.num_machines, self.config.num_machines)}
        )
        generator = JobGenerator(base)

        for attempt in self._attempt_loop():
            jobs = [generator.generate().lines for _ in range(self.config.num_jobs)]
            instance = _combine_jobs(jobs, self.config.num_machines)
            if instance is not None:
                return instance
        raise RuntimeError("exceeded max_resample_attempts without a valid combination")

    def _attempt_loop(self):
        cap = self.config.max_resample_attempts
        if cap is None:
            while True:
                yield None
        else:
            for i in range(cap):
                yield i


# ---------------------------------------------------------------------------
# Legacy shim — keeps `scripts/train_drl.py` working without changes
# ---------------------------------------------------------------------------


class CaseGenerator:
    """Backwards-compatible wrapper around :class:`InstanceGenerator`.

    Prefer :class:`InstanceGenerator` for new code; this shim exists so the
    training script can keep its old signature::

        case = CaseGenerator(num_jobs, num_machines, job_folder="data/jobs/job_with_mas_5/")
        env = IPPSEnv(case=case, env_paras=...)   # env calls case.get_case() per instance
    """

    def __init__(self, job_num: int = 5, machine_num: int = 15,
                 if_new_job: bool = False, job_folder: str | None = None) -> None:
        self.job_num = job_num
        self.machine_num = machine_num
        self.if_new_job = if_new_job
        self.job_folder = job_folder
        self._engine = InstanceGenerator(InstanceConfig(num_jobs=job_num, num_machines=machine_num))

    def get_case(self) -> List[str]:
        """Return the combined instance as a list of newline-stripped lines."""
        if self.if_new_job:
            return self._engine.from_fresh_jobs().lines
        if self.job_folder is None:
            raise ValueError("job_folder must be provided when if_new_job=False")
        return self._engine.from_pool(self.job_folder).lines


# ---------------------------------------------------------------------------
# CLI: bulk-generate IPPS instances from a job pool
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Bulk-generate IPPS instances from a job pool.")
    parser.add_argument("--job_folder", type=Path, required=True,
                        help="Folder of `..._mas_<N>.txt` job files")
    parser.add_argument("--out_dir", type=Path, default=Path("out"))
    parser.add_argument("--num_jobs", type=int, default=5)
    parser.add_argument("--num_machines", type=int, default=5)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen = InstanceGenerator(InstanceConfig(num_jobs=args.num_jobs, num_machines=args.num_machines))
    for i in range(args.count):
        inst = gen.from_pool(args.job_folder)
        name = (f"{args.num_jobs}_{args.num_machines}_problem_job_"
                f"{'_'.join(map(str, inst.source_job_ids or []))}.ipps")
        inst.save(args.out_dir / name)
        print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    _main()
