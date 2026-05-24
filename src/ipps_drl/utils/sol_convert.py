"""Solution format conversion helpers."""
import os

import numpy as np

from ipps_drl.env.load_data import load_ipps, nums_detec
from ipps_drl.utils.get_possible_set import get_comb_info
from ipps_drl.utils.utils import getAncestors, sort_schedule


def drl_to_ws(data_path, problem):
    """Convert a DRL ``.ippssol`` solution into a Gurobi warm-start ``.ws`` file."""
    pro_file = data_path + "/problem/" + problem
    with open(pro_file, 'r') as file_object:
        pro_lines = file_object.read().splitlines()
        num_jobs, _num_mas, _num_opes = nums_detec(pro_lines)
        tensor = load_ipps(pro_lines)

    matrix_ope_ma_adj = tensor[1].tolist()

    drl_sol_file = data_path + "/drl_solution/drl_sol_" + problem + 'sol'
    with open(drl_sol_file, 'r') as file_object:
        sol_lines = file_object.read().splitlines()
        makespan = round(float(sol_lines[0]))
        schedule = [line.split() for line in sol_lines[1:]]
        ope_ma_job_start_end = [
            [int(ope), int(ma), int(job), float(start), float(end)]
            for ope, ma, job, start, end in schedule
        ]

    id_combination, id_set_operation = get_comb_info(pro_file, num_jobs)

    folder = data_path + "/ws_solution/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    opes_i = {i: [] for i in range(num_jobs)}
    ma_j = {}
    complete_times = {}
    for ope_id, ma_id, job_id, _, _ in ope_ma_job_start_end:
        opes_i[job_id].append(ope_id)
        ma_j[ope_id] = ma_id

    comb_i = {
        i: key for i in range(num_jobs)
        for key, value in id_set_operation[i].items()
        if value == set(opes_i[i])
    }

    combination = {(i, h): h == comb_i[i] for i in range(num_jobs) for h in id_combination[i]}
    assignment = {
        (i, h, j, k): h == comb_i[i] and j in opes_i[i] and k == ma_j[j]
        for i in range(num_jobs)
        for h in id_combination[i]
        for j in id_set_operation[i][h]
        for k in np.where(matrix_ope_ma_adj[j])[0]
    }

    for ope_id, _, job_id, _, complete_time in ope_ma_job_start_end:
        complete_times[(job_id, comb_i[job_id], ope_id)] = complete_time

    ws_name = "ws_sol_" + problem.split('.')[0] + ".ws"
    with open(os.path.join(folder, ws_name), "w") as file:
        file.write(f"# Objective value {makespan}\nmakespan\t\t\t{makespan}\n")
        for key, value in combination.items():
            key_str = ','.join(map(str, key))
            file.write(f"combination({key_str})\t\t\t{1 if value else 0}\n")
        for key, value in assignment.items():
            key_str = ','.join(map(str, key))
            file.write(f"assignment({key_str})\t\t\t{1 if value else 0}\n")
        for key, value in complete_times.items():
            key_str = ','.join(map(str, key))
            file.write(f"complete_times({key_str})\t\t\t{value}\n")


def sort_sol(original_sol_lines, ope_pre_adj_batch):
    """Topologically re-sort a schedule using the operations' precedence matrix."""
    to_sol_lines = [[float(original_sol_lines[0])]]
    matrix_cal_cumul = getAncestors(ope_pre_adj_batch)
    to_sol_lines[1:] = sort_schedule(
        [list(map(float, line.split())) for line in original_sol_lines[1:]],
        matrix_cal_cumul,
    )
    return to_sol_lines
