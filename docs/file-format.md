# File format

## Problem — `.ipps`

Four sections.

### 1. Header

```
<num_jobs> <num_machines> <num_operations>
```

### 2. Graph (`out`)

Starts with a literal `out` line. Each subsequent line declares the outgoing edges of one operation:

```
<u> <v1> <v2> ...
```

means the edges `u → v1`, `u → v2`, … . An OR-connector is written by wrapping the OR-set in parentheses with no spaces:

```
1 (15,11,13)
```

means operation `1` has an OR fork — exactly one of `{15, 11, 13}` will eventually be picked.

### 3. Join (`in`)

Starts with a literal `in` line. Each line names the operations that join into one downstream operation:

```
4 (2,3)
```

means operations `2` and `3` are the tails of two OR branches that join at operation `4`.

### 4. Processing time (`info`)

Starts with a literal `info` line. Three kinds of rows:

- `<op_id> start` — the job's super-start node (no processing).
- `<op_id> end` — the job's super-end node (no processing).
- `<op_id> supernode` — an internal OR helper node (no processing).
- `<op_id> <n> <m_1> <t_1> <m_2> <t_2> ...` — operation `op_id` can be processed on `n` machines; machine `m_i` takes time `t_i`.

### Minimal example

A 1-job / 2-machine illustration:

```
1 2 5
out
0 1
1 (2,3)
2 4
3 4
in
4 (2,3)
info
0 start
1 1 1 3
2 1 2 5
3 1 1 4
4 end
```

The full bundled example lives at `data/example/problem.ipps` and is rendered below:

<img src="https://raw.githubusercontent.com/Lhongpei/ipps-drl/main/data/example/pic/job.png" alt="Job example" width="500">

## Solution — `.ippssol`

```
<total_makespan>
<op_id> <ma_id> <job_id> <start_time> <end_time>
<op_id> <ma_id> <job_id> <start_time> <end_time>
...
```

First line is the total makespan; every subsequent line is one scheduled operation.
