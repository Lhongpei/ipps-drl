import numpy as np
import os

from get_possible_set import get_comb_info
from env.load_data import nums_detec, load_ipps
from utils.utils import sort_schedule, getAncestors



def convert_sols_ws(data_path):
    pro_path = "{0}/problem/".format(data_path)
    drl_sol_path = "{0}/drl_solution/".format(data_path)
    pro_files = os.listdir(pro_path)
    drl_sol_files = os.listdir(drl_sol_path)
    for drl_sol in drl_sol_files:
        problem = drl_sol[len("drl_sol_"):]
        if problem not in pro_files:
            Warning("No corresponding problem.")
        drl_to_ws(data_path, problem)

def drl_to_ws(data_path, problem):
    # read problem file
    pro_file = data_path + "/problem/" + problem
    with open(pro_file, 'r') as file_object:
        pro_lines = file_object.read().splitlines()
        num_jobs, num_mas, num_opes = nums_detec(pro_lines)
        tensor = load_ipps(pro_lines)
    
    matrix_ope_ma_adj = tensor[1].tolist()

    # read drl solution file
    drl_sol_file = data_path + "/drl_solution/drl_sol_" + problem
    with open(drl_sol_file, 'r') as file_object:
        sol_lines = file_object.read().splitlines()
        makespan = round(float(sol_lines[0]))
        schedule = [line.split() for line in sol_lines[1:]]
        ope_ma_job_start_end = [[int(ope), int(ma), int(job), float(start), float(end)] for ope, ma, job, start, end in schedule]
    
    # get combination info
    id_combination, id_set_operation = get_comb_info(pro_file, num_jobs)

    folder = data_path + "/ws_solution/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # which combination for every job
    opes_i = {i :[] for i in range(num_jobs)}
    ma_j = {}

    # complete times
    complete_times = {}

    for pair in ope_ma_job_start_end:
        ope_id, ma_id, job_id, _, _ = pair
        opes_i[job_id].append(ope_id)
        ma_j[ope_id] = ma_id
        
    comb_i = {i: key for i in range(num_jobs) for key, value in id_set_operation[i].items() if value == set(opes_i[i])}

    # combination, assignment
    combination = {(i, h) : h == comb_i[i] for i in range(num_jobs) for h in id_combination[i]}
    assignment = {(i, h, j, k): h == comb_i[i] and j in opes_i[i] and k == ma_j[j] for i in range(num_jobs) for h in id_combination[i]
                  for j in id_set_operation[i][h] for k in np.where(matrix_ope_ma_adj[j])[0]}


    for pair in ope_ma_job_start_end:
        ope_id, _, job_id, _, complete_time = pair
        complete_times[(job_id, comb_i[job_id], ope_id)] = complete_time

    # output
    with open(os.path.join(folder, 'ws_sol_' + problem), "w") as file:
        file.write(f"# Objective value {makespan}\nmakespan\t\t\t{makespan}\n")
        for key, value in combination.items():
            key_str = ','.join(map(str, key))
            line = f"combination({key_str})\t\t\t{1 if value else 0}\n"
            file.write(line)
        for key, value in assignment.items():
            key_str = ','.join(map(str, key))
            line = f"assignment({key_str})\t\t\t{1 if value else 0}\n"
            file.write(line)
        for key, value in complete_times.items():
            key_str = ','.join(map(str, key))
            line = f"complete_times({key_str})\t\t\t{value}\n"
            file.write(line)

def order_correct(origin_sol_folder, pro_folder, to_sol_folder):
    if not os.path.exists(to_sol_folder):
        os.makedirs(to_sol_folder)
    for sol_file in os.listdir(origin_sol_folder):
        if not sol_file.endswith("optimal.txt"):
            continue
        prefix = sol_file[:-4]
        pro_file = prefix[8:-8] if prefix.endswith("_optimal") else prefix[8:]
        pro_path = os.path.join(pro_folder, pro_file)
        if not os.path.exists(pro_path):
            raise Warning("No problem file.")
        origin_sol_path = os.path.join(origin_sol_folder, sol_file)
        to_sol_path = os.path.join(to_sol_folder, sol_file)

        with open(pro_path, 'r') as file:
            pro_lines = file.read().splitlines()
        tensor = load_ipps(pro_lines)

        with open(origin_sol_path, 'r') as file:
            original_sol_lines = file.read().splitlines()

        to_sol_lines = []
        for i, line in enumerate(original_sol_lines):
            if i == 0:
                to_sol_lines.append(list(map(int, line.split())))
            else:
                line = list(map(int, line.split()))
                # job, ope, ma, start  ->  ope, ma, job, start, end
                new_line = [line[1], line[2], line[0], line[3]]
                new_line.append(line[3] + int(tensor[0][line[1], line[2]]))
                to_sol_lines.append(new_line)
        matrix_cal_cumul = getAncestors(tensor[2])
        to_sol_lines[1:] = sort_schedule(to_sol_lines[1:], matrix_cal_cumul)

        with open(to_sol_path, 'w') as file:
            for row in to_sol_lines:
                file.write(' '.join(map(str, row)) + '\n')

def sort_sols(origin_sol_folder, pro_folder, to_sol_folder):
    if not os.path.exists(to_sol_folder):
        os.makedirs(to_sol_folder)
    for sol_file in os.listdir(origin_sol_folder):
        prefix = sol_file[:-4]
        pro_file = (prefix[8:-8] if prefix.endswith("_optimal") else prefix[8:]) + ".txt"
        pro_path = os.path.join(pro_folder, pro_file)
        if not os.path.exists(pro_path):
            raise Warning("No problem file.")
        origin_sol_path = os.path.join(origin_sol_folder, sol_file)
        to_sol_path = os.path.join(to_sol_folder, sol_file)

        with open(pro_path, 'r') as file:
            pro_lines = file.read().splitlines()
        tensor = load_ipps(pro_lines)

        with open(origin_sol_path, 'r') as file:
            original_sol_lines = file.read().splitlines()
    
        to_sol_lines = sort_sol(original_sol_lines, tensor[2])
        
        with open(to_sol_path, 'w') as file:
            for row in to_sol_lines:
                file.write(' '.join(map(str, row)) + '\n')
        
def sort_sol(original_sol_lines, ope_pre_adj_batch):
    to_sol_lines = []
    to_sol_lines.append([float(original_sol_lines[0])])
    matrix_cal_cumul = getAncestors(ope_pre_adj_batch)
    to_sol_lines[1:] = sort_schedule([list(map(float, line.split())) for line in original_sol_lines[1:]], matrix_cal_cumul)
    return to_sol_lines


        


    

if __name__ == "__main__":
    origin_sol_folder = "IL_test/0505/solution"
    pro_folder = "IL_test/0505/problem"
    to_sol_folder = "IL_test/0505/correct_solution"
    sort_sols(origin_sol_folder, pro_folder, to_sol_folder)
