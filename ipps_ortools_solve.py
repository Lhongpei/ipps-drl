from ortools.sat.python import cp_model
from env.load_data import load_ipps
from utils.get_possible_set import read_ipps_data, get_comb_info
import numpy as np
import os
import pandas as pd
import time
import re
import argparse
def solve_ipps_with_ortools(id_job, id_machine, id_operation, id_set_operation, id_combination, process_time, machine_oper, ope_ma_adj, matrix_pre_proc, matrix_cal_cumul, drl_sol=None, 
                            save_path=None, only_comb = False, M = int(9e6), init_ub = int(1e4), time_limit = 3600, workers = 32):
    # Create model
    model = cp_model.CpModel()

    # Variable 1: Makespan
    makespan = model.NewIntVar(0, init_ub, 'makespan')

    # Variable 2: Job combination selection (Y)
    combination = {}
    for i in id_job:
        for h in id_combination[i]:
            combination[(i, h)] = model.NewBoolVar(f'combination_{i}_{h}')

    # Variable 3: Job assignment to machines (X)
    assignment = {}
    for i in id_job:
        for h in id_combination[i]:
            for j in id_set_operation[i][h]:
                for k in id_machine:
                    if (i, j, k) in ope_ma_adj.keys():
                        assignment[(i, h, j, k)] = model.NewBoolVar(f'assignment_{i}_{h}_{j}_{k}')

    # Variable 4: Operation precedence (Z)
    sequence = {}
    for i in id_job:
        for j1 in id_operation[i]:
            for j2 in id_operation[i]:
                sequence[(i, j1, j2)] = model.NewBoolVar(f'sequence_{i}_{j1}_{j2}')

    # Variable 5: Completion times of operations
    complete_times = {}
    for i in id_job:
        for h in id_combination[i]:
            for j in id_set_operation[i][h]:
                complete_times[(i, h, j)] = model.NewIntVar(0, init_ub, f'complete_times_{i}_{h}_{j}')

    # Auxiliary variable: Order on machine
    order_on_machine = {}
    for i1 in id_job:
        for i2 in id_job:
            for j1 in id_operation[i1]:
                for j2 in id_operation[i2]:
                    order_on_machine[(i1, j1, i2, j2)] = model.NewBoolVar(f'order_on_machine_{i1}_{j1}_{i2}_{j2}')

    # Add constraints
    # Constraint 1: Each job must select exactly one combination
    for i in id_job:
        model.Add(sum(combination[i, h] for h in id_combination[i]) == 1)

    # Constraint 2: Each operation is assigned to at most one machine
    for i in id_job:
        for h in id_combination[i]:
            for j in id_set_operation[i][h]:
                model.Add(sum(assignment[i, h, j, k] for k in machine_oper[(i, j)]) == combination[i, h])

    # Constraint 3: Combination feasibility
    for i in id_job:
        for h in id_combination[i]:
            for j in id_set_operation[i][h]:
                model.Add(combination[i, h] * M >= complete_times[i, h, j])

    # Constraint 4: Precedence constraints
    for i in id_job:
        for h in id_combination[i]:
            for j1 in id_set_operation[i][h]:
                for j2 in id_set_operation[i][h]:
                    if j1 != j2 and matrix_pre_proc[j1, j2] == 1:
                        model.Add(complete_times[i, h, j2] >= complete_times[i, h, j1] + sum(assignment[i, h, j2, k2] * process_time[(i, j2, k2)] for k2 in machine_oper[(i, j2)]))

    # Constraint 5: Sequence constraints
    for i in id_job:
        for j1 in id_operation[i]:
            for j2 in id_operation[i]:
                if j1 != j2 and matrix_cal_cumul[j1, j2] + matrix_cal_cumul[j2, j1] == 0:
                    model.Add(sequence[i, j1, j2] + sequence[i, j2, j1] == 1)

    # Constraint 6: Operation order
    for i in id_job:
        for h in id_combination[i]:
            for j1 in id_set_operation[i][h]:
                for j2 in id_set_operation[i][h]:
                    if j1 != j2:
                        model.Add(complete_times[i, h, j2] >= complete_times[i, h, j1] + sum(assignment[i, h, j2, k] * process_time[(i, j2, k)] for k in machine_oper[(i, j2)]) - M * (1 - sequence[i, j1, j2]))

    # Constraint 7 & 8: Machine order
    for i1 in id_job:
        for i2 in id_job:
            for h1 in id_combination[i1]:
                for h2 in id_combination[i2]:
                    for j1 in id_set_operation[i1][h1]:
                        for j2 in id_set_operation[i2][h2]:
                            for k1 in machine_oper[(i1, j1)]:
                                for k2 in machine_oper[(i2, j2)]:
                                    if i1 != i2 and k1 == k2:
                                        model.Add(complete_times[i2, h2, j2] >= complete_times[i1, h1, j1] + assignment[i2, h2, j2, k2] * process_time[(i2, j2, k2)] - M * (1 - order_on_machine[(i1, j1, i2, j2)]) - M * (2 - assignment[i1, h1, j1, k1] - assignment[i2, h2, j2, k2]))
                                        model.Add(complete_times[i1, h1, j1] >= complete_times[i2, h2, j2] + assignment[i1, h1, j1, k1] * process_time[(i1, j1, k1)] - M * order_on_machine[(i1, j1, i2, j2)] - M * (2 - assignment[i1, h1, j1, k1] - assignment[i2, h2, j2, k2]))

    # Constraint 9: Define makespan
    for i in id_job:
        for h in id_combination[i]:
            for j in id_set_operation[i][h]:
                model.Add(makespan >= complete_times[i, h, j])

    # Objective function: Minimize makespan
    model.Minimize(makespan)

    # Warm start with DRL solution if provided
    if drl_sol:
        ortools_warm_start(model, makespan, combination, assignment, complete_times, drl_sol, only_comb)

    # Solve the model
    print("Solving the model...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True  
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)
    solve_time = solver.WallTime()

    # Output the results
    if status == cp_model.OPTIMAL:
        print("Optimal solution found.")
    elif status == cp_model.INFEASIBLE:
        print("No feasible solution.")
    else:
        print("Solver status:", status)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Operation schedule for each job:")
        if save_path:
            schedule = []
        for i in id_job:
            for h in id_combination[i]:
                for j in id_set_operation[i][h]:
                    for k in machine_oper[(i, j)]:
                        if solver.BooleanValue(assignment[(i, h, j, k)]):  # If the operation is assigned to the machine
                            start = solver.Value(complete_times[(i, h, j)]) - process_time[(i, j, k)]
                            end = solver.Value(complete_times[(i, h, j)])
                            assignment_value = solver.Value(assignment[(i, h, j, k)])
                            print(f"Job {i}, Operation {j}, Machine {k}: Start time: {start}, End time: {end}, Assignment: {assignment_value}")
                            schedule.append([j, k, i, start, end])
        if save_path:
            if status == cp_model.OPTIMAL:
                split_path = save_path.split('.')
                save_path = split_path[0] + '_optimal.' + split_path[1]
            schedule = sorted(schedule, key=lambda x: x[3])
            with open(save_path, 'w') as file:
                file.write(str(solver.Value(makespan)) + '\n')
                for row in schedule:
                    file.write(' '.join(map(str, row)) + '\n')

        print("Total makespan:", solver.Value(makespan))
        
    else:
        print("No feasible solution found.")
        split_path = save_path.split('.')
        save_path = split_path[0] + '_infeasible.' + split_path[1]
        with open(save_path, 'w') as file:
            file.write("Infeasible")
        return solve_time, None
    return solve_time, solver.Value(makespan)

def ortools_warm_start(model, makespan, combination, assignment, complete_times, ws_sol, only_comb = False):
    with open(ws_sol, 'r') as file:
        next(file)
        for line in file:
            # Remove trailing newline and extra spaces
            key_str, value = line.strip().split()
            if key_str == "makespan":
                var = makespan
                model.AddHint(var, int(value))
            else:
                match = re.match(r'(\w+)\(([\d,]+)\)', key_str)
                var_name = match.group(1)
                index = tuple(map(int, match.group(2).split(',')))
                var = locals()[var_name][index]
                if only_comb and var_name != 'combination':
                    continue
                model.AddHint(var, int(float(value)))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--file_folder', default='data_test/0405/problem', help='File folder path')
    parser.add_argument('--save_folder', default='data_test/0405/solution', help='Save folder path')
    parser.add_argument('--ws_folder', default=None, help='Warm Start folder path, default None that means no warm start')
    parser.add_argument('--log_folder', default='solver_log', help='Log folder path')
    parser.add_argument('--log', type=bool, default=False, help='Use log?')
    parser.add_argument('--only_comb', type=bool, default=False, help='Only warm start combination?')
    parser.add_argument('--big_M', type=int, default=int(9e6), help='Big M value')
    parser.add_argument('--init_ub', type=int, default=int(1e4), help='Initial UB')
    parser.add_argument('--time_limit', type=int, default=3600, help='Time limit')
    parser.add_argument('--workers', type=int, default=32, help='Number of workers')

    args = parser.parse_args()

    file_folder = args.file_folder
    save_folder = args.save_folder
    ws_folder = args.ws_folder
    log_folder = args.log_folder
    log = args.log
    only_comb = args.only_comb
    big_M = args.big_M
    init_ub = args.init_ub
    time_limit = args.time_limit
    workers = args.workers
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if log and not os.path.exists(log_folder):
        os.makedirs(log_folder)
    files = os.listdir(file_folder)
    files.sort(key=lambda x: x[:-4])
    times = []
    makespans = []
    result = pd.DataFrame()
    result['file'] = files

    for file in files:
        # try:
        print(file)

        file_path = os.path.join(file_folder, file)
        save_path = os.path.join(save_folder, "o2d_sol_" + file)
        ws_path = os.path.join(ws_folder, "ws_sol_" + file) if ws_folder is not None else ''

        # check if the solution file already exists, if so, skip this file.
        split_path = save_path.split('.')
        save_path_optimal = split_path[0] + '_optimal.' + split_path[1]
        save_path_infeasible = split_path[0] + '_infeasible.' + split_path[1]
        save_path_error = split_path[0] + '_error.' + split_path[1]
        if os.path.exists(save_path) or os.path.exists(save_path_optimal) or \
            os.path.exists(save_path_infeasible) or os.path.exists(save_path_error):
            print(f"{save_path} already exists. Skipping this file.")
            continue

        with open(file_path, 'r') as file:
            lines = file.read().splitlines()

        matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, \
            opes_appertain, num_ope_biases, nums_ope, matrix_cal_cumul, \
            ope_or, ope_req_num, ope_eligible, _, _,_ = load_ipps(lines)

        first_line = lines[0].strip().split()
        num_jobs, num_machines, num_operation = int(first_line[0]), int(first_line[1]), int(first_line[2])

        id_combination, id_set_operation = get_comb_info(file_path, num_jobs)

        num_operations = nums_ope
        operations_start = num_ope_biases

        id_job = [i for i in range(num_jobs)]
        id_machine = [i for i in range(num_machines)]
        id_operation = {i: [j for j in range(operations_start[i], operations_start[i] + num_operations[i])]
                        for i in range(0, num_jobs)}

        machine_oper, process_time, ope_ma_adj = read_ipps_data(file_path, matrix_proc_time,
                                                                matrix_ope_ma_adj, id_operation)

        solve_time, makespan = solve_ipps_with_ortools(id_job, id_machine, id_operation, id_set_operation, id_combination,
                                process_time, machine_oper, ope_ma_adj, matrix_pre_proc, matrix_cal_cumul,
                                save_path=save_path, drl_sol=ws_path# if os.path.exists(ws_path) else None
                                , only_comb = only_comb, M = big_M, init_ub = init_ub, time_limit = time_limit, workers = workers)
        times.append(solve_time)
        makespans.append(makespan)
    if log:
        result['time'] = times
        result['makespan'] = makespans
        result.to_csv(f'{log_folder}/result_{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}.csv', index=False)

