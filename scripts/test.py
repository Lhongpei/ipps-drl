"""Evaluate trained DRL checkpoints on a directory of .ipps instances.

Thin wrapper around :class:`ipps_drl.inference.InferenceEngine`. For every
checkpoint under ``--checkpoint_dir`` (default ``checkpoints/``) and every
instance under ``--data_dir``, runs DRL inference (``greedy`` or ``sampling``)
and writes a per-checkpoint CSV under ``--save_dir``.

Examples::

    # Greedy evaluation across the bundled Kim benchmark
    python scripts/test.py --method greedy

    # Sampling (DRL-S) with 25 parallel rollouts, 2 retries each
    python scripts/test.py --method sampling --num_sample 25 --num_average 2

    # Pick a single checkpoint and a single instance
    python scripts/test.py --checkpoint checkpoints/0605.pt \
                          --data_dir data/test/kim/problem
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ipps_drl.inference import InferenceEngine


def _setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _list_checkpoints(checkpoint: str | None, checkpoint_dir: Path) -> list[Path]:
    if checkpoint is not None:
        return [Path(checkpoint)]
    return sorted(checkpoint_dir.glob("*.pt"))


def _save_solution(out_dir: Path, problem_name: str, makespan: float, schedule) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{makespan}"]
    for row in schedule:
        lines.append(f"{int(row[0])} {int(row[1])} {int(row[2])} {row[3]} {row[4]}")
    (out_dir / f"drl_sol_{problem_name}.sol").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/test/kim/problem"),
                        help="Directory of .ipps problem files")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a single .pt checkpoint (overrides --checkpoint_dir)")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"),
                        help="Directory of .pt checkpoints (used when --checkpoint is not given)")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="Output directory (default: save/test_<timestamp>)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--method", choices=["greedy", "sampling"], default="greedy")
    parser.add_argument("--num_sample", type=int, default=25,
                        help="Parallel sample size for --method=sampling (DRL-S)")
    parser.add_argument("--num_average", type=int, default=1,
                        help="Number of independent rollouts; keeps the best across them")
    parser.add_argument("--save_solutions", action="store_true",
                        help="Write each best schedule to <save_dir>/solutions/")
    parser.add_argument("--num_ins", type=int, default=None,
                        help="Cap on number of instances (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _setup_seed(args.seed)

    problem_files = sorted(args.data_dir.glob("*.ipps")) + sorted(args.data_dir.glob("*.txt"))
    if args.num_ins is not None:
        problem_files = problem_files[:args.num_ins]
    if not problem_files:
        raise SystemExit(f"No .ipps/.txt instances found in {args.data_dir}")

    checkpoints = _list_checkpoints(args.checkpoint, args.checkpoint_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found (looked in {args.checkpoint_dir})")

    save_dir = args.save_dir or Path(f"save/test_{time.strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"data_dir   : {args.data_dir} ({len(problem_files)} instances)")
    print(f"checkpoints: {[c.name for c in checkpoints]}")
    print(f"method     : {args.method}"
          + (f"  (num_sample={args.num_sample}, num_average={args.num_average})"
             if args.method == "sampling" else ""))
    print(f"save_dir   : {save_dir}")

    extra: dict = {}
    if args.method == "sampling":
        extra = {"num_sample": args.num_sample, "num_average": args.num_average}

    start = time.time()
    for ckpt in checkpoints:
        engine = InferenceEngine(checkpoint=str(ckpt), device=args.device)
        rows = []
        for prob in problem_files:
            r = engine.solve(str(prob), method=args.method, **extra)
            print(f"  {ckpt.name}  {prob.name}  makespan={r.makespan:.2f}  time={r.wall_time_s:.2f}s")
            rows.append({
                "file_name": prob.name,
                "makespan": r.makespan,
                "wall_time_s": r.wall_time_s,
            })
            if args.save_solutions and r.schedule is not None:
                _save_solution(save_dir / "solutions", prob.stem, r.makespan, r.schedule)
        pd.DataFrame(rows).to_csv(save_dir / f"result_{ckpt.stem}.csv", index=False)
    print(f"\ntotal time: {time.time() - start:.1f}s — results written to {save_dir}")


if __name__ == "__main__":
    main()
