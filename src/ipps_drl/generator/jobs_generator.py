"""Random job generation for IPPS instances.

Public API:
    - :class:`JobConfig`   — knobs for one job (DAG shape, OR/AND splits, machines, ...).
    - :class:`Job`         — generated job (lines + parsed metadata + ``save()``).
    - :class:`JobGenerator` — ``JobGenerator(config).generate() -> Job``.
    - :func:`jobs_generator` — backwards-compatible flat function returning a list of
      ``.txt`` lines (the format used by ``data/jobs/job_with_mas_*/``).

Example::

    from ipps_drl.generator import JobGenerator, JobConfig

    job = JobGenerator(JobConfig(machine_range=(5, 5))).generate()
    job.save("data/jobs/job_with_mas_5/job_42")  # writes "..._mas_5.txt"
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass
class JobConfig:
    """Configuration for generating one job.

    All fields have sensible defaults; override only what you need.
    """

    # Machines
    machine_range: Tuple[int, int] = (4, 10)   # min, max machines available
    mas_p: float = 0.5                         # P(a machine can process a given op)
    time_range: Tuple[int, int] = (100, 500)   # per-op mean processing time
    time_bias: Tuple[int, int] = (3, 5)        # per-machine jitter around the mean

    # DAG skeleton
    max_out: int = 2                           # DAG max out-degree
    alpha: float = 1.0                         # DAG shape parameter
    beta: float = 1.0                          # DAG regularity parameter

    # Operation count
    randomize_opes: bool = False               # if True, draw ope_num from `ope_range`
    ope_num: int = 10                          # base operation count (if not randomised)
    ope_range: Tuple[int, int] = (4, 20)       # range to draw ope_num from
    total_ope_range: Tuple[int, int] = (10, 100)   # bounds for the final total op count

    # OR / AND structure
    or_num: int = 3                            # absolute number of OR splits
    or_p: float = 0.0                          # overrides `or_num` when > 0: ratio of OR
    or_road_num: int = 3                       # number of OR branches per split
    ope_num_orpath: int = 3                    # max ops per OR branch
    and_p: float = 0.3                         # P(insert AND branches inside an OR branch)
    and_road_num: int = 3                      # number of parallel AND branches
    ope_num_andpath: int = 3                   # max ops per AND branch


@dataclass
class Job:
    """One generated job (as it would be written to disk).

    Attributes:
        lines: Text lines exactly as they would appear in a ``.txt`` file (no trailing newlines).
        num_machines: Number of machines actually used by this job (header field).
        num_opes: Total number of operations including supernodes (header field).
    """

    lines: List[str]
    num_machines: int
    num_opes: int

    def to_text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def save(self, path: str | Path) -> Path:
        """Write the job to disk. ``_mas_<num_machines>.txt`` is appended to ``path``.

        This matches the format used by ``data/jobs/job_with_mas_<N>/job_<id>_mas_<N>.txt``.
        """
        out = Path(f"{path}_mas_{self.num_machines}.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_text())
        return out


# ---------------------------------------------------------------------------
# Internal DAG / OR-AND helpers (algorithm preserved verbatim)
# ---------------------------------------------------------------------------


def _dag_generate(n: int, max_out: int, alpha: float, beta: float):
    """Random layered DAG with `n` internal ops + a super-start (0) and super-end (n+1)."""
    length = math.floor(math.sqrt(n) / alpha)
    mean_value = n / length
    random_num = np.random.normal(loc=mean_value, scale=beta, size=(length,))

    position = {0: (0, 4), n: (10, 4)}
    generate_num = 0
    dag_num = 1
    dag_list: List[list] = []
    for i in range(len(random_num)):
        dag_list.append([])
        for j in range(math.ceil(float(random_num[i]))):
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != n:
        if generate_num < n:
            for _ in range(n - generate_num):
                index = random.randrange(0, length, 1)
                dag_list[index].append(len(dag_list[index]))
        else:
            i = 0
            while i < generate_num - n:
                index = random.randrange(0, length, 1)
                if len(dag_list[index]) <= 1:
                    continue
                del dag_list[index][-1]
                i += 1

    dag_list_update: List[list] = []
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num, dag_num + len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3 * (i + 1), pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position[0] = (0, max_pos / 2)
        position[n + 1] = (3 * (length + 1), max_pos / 2)

    into_degree = [0] * n
    out_degree = [0] * n
    edges: List[tuple] = []
    pred = 0
    for i in range(length - 1):
        sample_list = list(range(len(dag_list_update[i + 1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1, max_out + 1, 1)
            od = len(dag_list_update[i + 1]) if len(dag_list_update[i + 1]) < od else od
            bridge = random.sample(sample_list, od)
            for k in bridge:
                edges.append((dag_list_update[i][j], dag_list_update[i + 1][k]))
                into_degree[pred + len(dag_list_update[i]) + k] += 1
                out_degree[pred + j] += 1
        pred += len(dag_list_update[i])

    for node, id_ in enumerate(into_degree):
        if id_ == 0:
            edges.append((0, node + 1))
            into_degree[node] += 1
    for node, od in enumerate(out_degree):
        if od == 0:
            edges.append((node + 1, n + 1))
            out_degree[node] += 1

    return edges, into_degree, out_degree, position


def _add_and_paths(edges, ope_num, start_node, end_node, road_num=3, ope_num_andpath=3):
    edge_dict: dict = {}
    and_road: list = []
    road_num = random.randint(2, road_num)
    for _ in range(road_num):
        current_node = start_node
        operations = random.randint(1, ope_num_andpath)
        for _ in range(operations):
            ope_num += 1
            new_node = ope_num - 1
            edges.append((current_node, new_node))
            and_road.append(new_node)
            current_node = new_node
        edges.append((current_node, end_node))
    for key, value in edges:
        key = int(key)
        value = int(value)
        edge_dict.setdefault(key, set()).add(value)
    return edge_dict, edges, ope_num, and_road


def _add_or_paths(edges, ope_num, start_node, join_node, join_or, or_road,
                  road_num=3, ope_num_orpath=3, and_road_num=3, ope_num_andpath=3,
                  and_p=0.3, add_super=False):
    super_set: set = set()
    or_road[start_node] = {}
    edge_dict: dict = {}
    road_num = random.randint(2, road_num)
    if add_super:
        end_node = ope_num
        ope_num += 1
        edges.append((end_node, join_node))
        super_set.add(end_node)
        or_node = join_or[join_node]
        for i in or_road[or_node]:
            if or_road[or_node][i][-1] == start_node:
                or_road[or_node][i].append(end_node)
    else:
        end_node = join_node
    for i in range(road_num):
        current_node = start_node
        operations = random.randint(1, ope_num_orpath)
        or_road[start_node][i] = []
        and_ope: list = []
        for _ in range(operations):
            ope_num += 1
            new_node = ope_num - 1
            or_road[start_node][i].append(new_node)
            edges.append((current_node, new_node))
            if current_node != start_node and random.random() <= and_p:
                edge_dict, edges, ope_num, and_road = _add_and_paths(
                    edges, ope_num, current_node, new_node, and_road_num, ope_num_andpath
                )
                and_ope.extend(and_road)
            current_node = new_node
        edges.append((current_node, end_node))
        for ope in and_ope:
            or_road[start_node][i].insert(-1, ope)
    for key, value in edges:
        key = int(key)
        value = int(value)
        edge_dict.setdefault(key, set()).add(value)
    return edge_dict, edges, ope_num, or_road, super_set


# Predecessor lookup is exposed for users that want to walk the generated DAG.
def search_for_predecessor(node, edges):
    """Return the predecessor node IDs of ``node`` in ``edges``."""
    pred_map: dict = {}
    if node == 'Start':
        raise ValueError("'Start' node has no predecessor")
    for u, v in edges:
        pred_map.setdefault(v, []).append(u)
    return pred_map[node]


# ---------------------------------------------------------------------------
# The actual generator
# ---------------------------------------------------------------------------


class JobGenerator:
    """Sample one :class:`Job` per :meth:`generate` call from the given :class:`JobConfig`."""

    def __init__(self, config: JobConfig | None = None) -> None:
        self.config = config or JobConfig()

    def generate(self) -> Job:
        cfg = self.config
        while True:
            # ---- counts
            if cfg.randomize_opes:
                ope_num = random.randint(cfg.ope_range[0], cfg.ope_range[1])
            else:
                ope_num = cfg.ope_num
            total_ope_lb, total_ope_ub = cfg.total_ope_range
            machine = random.randint(cfg.machine_range[0], cfg.machine_range[1])

            # ---- DAG skeleton
            edges, _in_deg, _out_deg, _pos = _dag_generate(
                ope_num, cfg.max_out, cfg.alpha, cfg.beta
            )
            super_end = ope_num + 1
            ope_num += 2

            node_dict = {op: op for op in range(ope_num)}
            edge_dict: dict = {}
            for u, v in edges:
                edge_dict.setdefault(int(u), set()).add(int(v))

            # ---- inject OR splits
            or_num = int(cfg.or_p * ope_num) if cfg.or_p else cfg.or_num
            or_road: dict = {}
            join_or: dict = {}
            or_join: dict = {}
            or_and: set = set()
            super_set: set = set()

            for _ in range(or_num):
                operation = random.choice(range(ope_num))
                while operation == super_end or operation in or_and:
                    operation = random.choice(range(ope_num))
                end_node = random.choice(list(edge_dict[operation]))
                or_and.add(operation)

                if end_node not in join_or:
                    join_or[end_node] = operation
                    or_join[operation] = end_node
                    add_super = False
                else:
                    add_super = True
                    or_join[operation] = ope_num
                    join_or[ope_num] = operation

                edges = [(k, v) for k, v in edges if not (k == operation and v == end_node)]
                edge_dict, edges, ope_num, or_road, super_node = _add_or_paths(
                    edges, ope_num, operation, end_node, join_or, or_road,
                    cfg.or_road_num, cfg.ope_num_orpath,
                    cfg.and_road_num, cfg.ope_num_andpath, cfg.and_p, add_super,
                )
                super_set.update(super_node)

            # ---- re-roll if `total_ope_range` not respected
            if cfg.randomize_opes and not (total_ope_lb + 2 <= ope_num <= total_ope_ub + 2):
                continue

            for op in range(super_end + 1, ope_num):
                node_dict[op] = op - 1
            node_dict[super_end] = ope_num - 1

            # ---- build the `out` / `in` / `info` sections
            out_info = ['out']
            or_next: dict = {}
            for operation in range(ope_num):
                if operation == super_end:
                    continue
                if operation not in or_road:
                    strs = ' '.join(str(node_dict[i]) for i in edge_dict[operation])
                    out_info.append(f"{node_dict[operation]} {strs}")
                    continue
                selected_ope: set = set()
                for i, _ in or_road[operation].items():
                    selected_ope.add(or_road[operation][i][0])
                not_selected_ope = list(set(edge_dict[operation]) - set(selected_ope))
                strs = ','.join(str(node_dict[i]) for i in selected_ope)
                strs = f"({strs}) "
                strs = strs + " ".join(str(node_dict[i]) for i in not_selected_ope)
                out_info.append(f"{node_dict[operation]} {strs}")
                or_next[operation] = selected_ope

            in_info = ['in']
            for join, or_node in join_or.items():
                or_pre: set = set()
                for i, _ in or_road[or_node].items():
                    or_pre.add(or_road[or_node][i][-1])
                in_str = "(" + ','.join(str(node_dict[i]) for i in or_pre) + ")"
                in_info.append(f"{node_dict[join]} {in_str}")

            time_lb, time_ub = cfg.time_range
            bias_lb, bias_ub = cfg.time_bias
            mas_ope = ['info', '0 start']
            used_mas: set = set()
            info_end = f'{node_dict[super_end]} end'
            for operation in range(1, ope_num):
                if operation in super_set:
                    mas_ope.append(f'{node_dict[operation]} supernode')
                    continue
                if operation == super_end:
                    continue
                mas: list = []
                while len(mas) == 0:
                    mean_time = random.randint(time_lb, time_ub)
                    bias = random.randint(bias_lb, bias_ub)
                    for mas_idx in range(1, machine + 1):
                        if random.random() < cfg.mas_p:
                            t = random.randint(mean_time - bias, mean_time + bias)
                            mas.append(f' {mas_idx} {t}')
                            used_mas.add(mas_idx)
                mas_ope.append(f'{node_dict[operation]} {len(mas)}' + ''.join(mas))

            # All machines must be used; otherwise re-roll.
            if len(used_mas) != machine:
                continue
            mas_ope.append(info_end)

            header = f"1 {len(used_mas)} {ope_num}"
            lines = [header, *out_info, *in_info, *mas_ope]
            return Job(lines=lines, num_machines=len(used_mas), num_opes=ope_num)


# ---------------------------------------------------------------------------
# Back-compat flat function (returns lines; used by InstanceGenerator)
# ---------------------------------------------------------------------------


def jobs_generator(
    mode: str = 'default',
    machine_range: Tuple[int, int] = (4, 10),
    mas_p: float = 0.5,
    or_p: float = 0.0,
    or_num: int = 3,
    ope: bool = False,
    ope_num: int = 10,
    ope_range: Tuple[int, int] = (4, 20),
    total_ope_range: Tuple[int, int] = (10, 100),
    time_range: Tuple[int, int] = (100, 500),
    time_bias: Tuple[int, int] = (3, 5),
    max_out: int = 2,
    alpha: float = 1.0,
    beta: float = 1.0,
    road_num: int = 3,
    ope_num_orpath: int = 3,
    and_road_num: int = 3,
    ope_num_andpath: int = 3,
    and_p: float = 0.3,
    save: bool = False,
    path: str | None = None,
) -> List[str]:
    """Legacy entry point — prefer :class:`JobGenerator` for new code.

    Returns the same list of newline-stripped text lines as before. ``mode`` is
    accepted but ignored (only ``'default'`` was ever exercised by callers).
    """
    del mode  # unused
    cfg = JobConfig(
        machine_range=machine_range, mas_p=mas_p, time_range=time_range, time_bias=time_bias,
        max_out=max_out, alpha=alpha, beta=beta,
        randomize_opes=ope, ope_num=ope_num, ope_range=ope_range, total_ope_range=total_ope_range,
        or_num=or_num, or_p=or_p,
        or_road_num=road_num, ope_num_orpath=ope_num_orpath,
        and_road_num=and_road_num, ope_num_andpath=ope_num_andpath, and_p=and_p,
    )
    job = JobGenerator(cfg).generate()
    if save and path is not None:
        job.save(path)
    return job.lines


# ---------------------------------------------------------------------------
# CLI: bulk-generate a pool of job files (replaces the old `__main__` block)
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Bulk-generate random job files for the IPPS pool.")
    parser.add_argument("--out_dir", type=Path, default=Path("data/jobs/job_with_mas_5"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--ope_num", type=int, default=10)
    parser.add_argument("--max_or", type=int, default=3, help="Each job gets a random or_num in [0, max_or]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for i in tqdm(range(args.count), desc=f"jobs->{args.out_dir}"):
        cfg = JobConfig(
            machine_range=(args.machines, args.machines),
            ope_num=args.ope_num,
            or_num=random.randint(0, args.max_or),
        )
        JobGenerator(cfg).generate().save(args.out_dir / f"job_{i}")


if __name__ == "__main__":
    _main()
