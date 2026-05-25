"""Batch MCTS evaluation over a directory of .ipps instances.

Thin wrapper around :class:`ipps_drl.inference.InferenceEngine`. Loads every
checkpoint under ``checkpoints/`` (or just the ``--checkpoint`` you specify),
runs MCTS on every instance in ``--data_dir``, and writes a CSV summary +
per-instance search-trace CSVs into ``--save_dir``.
"""
from __future__ import annotations

import argparse
import pickle
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ipps_drl.inference import InferenceEngine


# Hard-coded Kim benchmark lower bounds (problem01 .. problem24).
_KIM_LB = {
    1: 427, 2: 343, 3: 344, 4: 306, 5: 318, 6: 427,
    7: 372, 8: 343, 9: 427, 10: 427, 11: 344, 12: 318,
    13: 427, 14: 372, 15: 427, 16: 427, 17: 344, 18: 318,
    19: 427, 20: 372, 21: 427, 22: 427, 23: 372, 24: 427,
}


def _setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _list_checkpoints(checkpoint: str | None) -> list[Path]:
    if checkpoint is not None:
        return [Path(checkpoint)]
    return sorted(Path("checkpoints").glob("*.pt"))


def _kim_lower_bound(problem_path: str) -> float:
    m = re.search(r"problem(\d+)", problem_path)
    if not m:
        return -1
    return _KIM_LB.get(int(m.group(1)), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/test/kim/problem",
                        help="Directory of .ipps problem files")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a single .pt checkpoint; defaults to all under checkpoints/")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory (default: save/mcts_<timestamp>)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--time_limit", type=float, default=60.0,
                        help="MCTS time budget per instance, in seconds")
    parser.add_argument("--exploration", type=float, default=5.0,
                        help="UCT exploration constant")
    parser.add_argument("--use_kim_lb", action="store_true",
                        help="Use the hard-coded Kim benchmark lower bounds")
    parser.add_argument("--strong", action="store_true",
                        help="Enable strong-mode MCTS (AlphaZero PUCT + max-reward selection "
                             "+ root Dirichlet noise + rollout cache). Recommended on hard "
                             "instances; see ipps_drl/inference/mcts.py module docstring.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _setup_seed(args.seed)

    data_dir = Path(args.data_dir)
    problem_files = sorted(data_dir.glob("*.ipps")) + sorted(data_dir.glob("*.txt"))
    if not problem_files:
        raise SystemExit(f"No instances found in {data_dir}")

    checkpoints = _list_checkpoints(args.checkpoint)
    if not checkpoints:
        raise SystemExit("No checkpoints found")

    save_dir = Path(args.save_dir or f"save/mcts_{time.strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"data_dir={data_dir} ({len(problem_files)} instances)")
    print(f"checkpoints={[c.name for c in checkpoints]}")
    print(f"save_dir={save_dir}")

    summary_rows = []
    action_dict: dict[str, object] = {}

    for ckpt in checkpoints:
        engine = InferenceEngine(checkpoint=str(ckpt), device=args.device)
        for prob in problem_files:
            lb = _kim_lower_bound(str(prob)) if args.use_kim_lb else -1
            result = engine.solve(
                str(prob),
                method="mcts",
                time_limit=args.time_limit,
                exploration=args.exploration,
                lower_bound=lb,
                strong=args.strong,
            )
            print(f"  {ckpt.name}  {prob.name}  "
                  f"makespan={result.makespan:.2f}  time={result.wall_time_s:.2f}s")
            summary_rows.append({
                "problem": prob.name,
                "checkpoint": ckpt.name,
                "makespan": result.makespan,
                "wall_time_s": result.wall_time_s,
                "lower_bound": lb,
            })
            key = f"{prob.name}__{ckpt.name}"
            action_dict[key] = result.extras.get("action_list")
            trace = pd.DataFrame({
                "best_makespan": result.extras.get("best_makespan_trace") or [],
                "round_makespan": result.extras.get("round_makespan_trace") or [],
            })
            trace.to_csv(save_dir / f"{key}_search_trace.csv",
                         index_label="iteration", encoding="utf-8-sig")

    pd.DataFrame(summary_rows).to_csv(save_dir / "result_summary.csv", index=False)
    with open(save_dir / "action_dict.pkl", "wb") as f:
        pickle.dump(action_dict, f)
    print(f"\nResults written to {save_dir}")


if __name__ == "__main__":
    main()
