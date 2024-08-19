import gurobipy as gp
from gurobipy import GRB
import os
from env.load_data import load_ipps
from utils.get_possible_set import read_ipps_data, get_comb_info

def solve_ipps_with_gurobi(id_job, id_machine, id_operation, id_set_operation, id_combination, process_time, machine_oper, ope_ma_adj, matrix_pre_proc, matrix_cal_cumul, drl_sol=None,save_path=None,only_comb=False):
    # create a new model
    model = gp.Model("IPPS")

    model.setParam('LogToConsole', 1)
    model.setParam('TimeLimit', 1800) 
    print('Adding variables')

    # Variable 1: makespan
    makespan = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="makespan")

    #Variable 2: job combination selection
    combination = model.addVars(
        [(i, h) for i in id_job for h in id_combination[i]], vtype=GRB.BINARY, name="combination")

    #Variable 3: operation assignment, whether to be processed on a machine
    assignment = model.addVars([(i, h, j, k) for i in id_job for h in id_combination[i] for j in id_set_operation[i][h]
                                for k in id_machine if (i, j, k) in ope_ma_adj.keys()], vtype=GRB.BINARY, name="assignment")

    #Variable 4: Oij1 is processed directly or indirectly before Oij2
    sequence = model.addVars([(i, j1, j2) for i in id_job for j1 in id_operation[i]
                             for j2 in id_operation[i]], vtype=GRB.BINARY, name="sequence")

    # Variable 5: completion time of Oihj
    complete_times = model.addVars([(i, h, j) for i in id_job for h in id_combination[i]
                                   for j in id_set_operation[i][h]], vtype=GRB.CONTINUOUS, lb=0, name="complete_times")

    # Variable 6: Oi1j1 processed before Oi2j2 on a machine
    order_on_machine = model.addVars([(i1, j1, i2, j2) for i1 in id_job for i2 in id_job for j1 in id_operation[i1]
                                     for j2 in id_operation[i2]], vtype=GRB.BINARY, name="order_on_machine")

    print('Adding constraints')

    # Constraint 1: Each job can only be assigned to one combination
    model.addConstrs((gp.quicksum(combination[i, h] for h in id_combination[i]) == 1 for i in id_job), name="comb_job")

    # Constraint 2: An operation can only be assigned to one machine
    model.addConstrs((gp.quicksum(assignment[i, h, j, k] for k in machine_oper[(i, j)]) == combination[i, h]
                      for i in id_job for h in id_combination[i] for j in id_set_operation[i][h]), name="comb_ma")

    # Constraint 3
    model.addConstrs((30000 * combination[i, h] >= complete_times[i, h, j] for i in id_job
                      for h in id_combination[i] for j in id_set_operation[i][h]), name="comb_time")

    # Constraint 4
    model.addConstrs((complete_times[i, h, j2] >= complete_times[i, h, j1] + gp.quicksum(assignment[i, h, j2, k2] * process_time[(i, j2, k2)]
                     for k2 in machine_oper[(i, j2)]) for i in id_job for h in id_combination[i] for j1 in id_set_operation[i][h] for j2 in id_set_operation[i][h] if (j1 != j2) and matrix_pre_proc[j1, j2] == 1), name="seq_time_1")

    # Constraint 5
    model.addConstrs((sequence[i, j1, j2] + sequence[i, j2, j1] == 1 for i in id_job for j1 in id_operation[i]
                      for j2 in id_operation[i] if (j1 != j2) and (matrix_cal_cumul[j1, j2] + matrix_cal_cumul[j2, j1] == 0)), name="seq_time_2")

    # Constraint 6
    model.addConstrs((complete_times[i, h, j2] >= complete_times[i, h, j1] + gp.quicksum(assignment[i, h, j2, k] * process_time[(i, j2, k)] for k in machine_oper[(i, j2)])
                      - 30000 * (1 - sequence[i, j1, j2]) for i in id_job for h in id_combination[i] for j1 in id_set_operation[i][h] for j2 in id_set_operation[i][h] if (j1 != j2)), name="seq_time_3")

    # Constraint 7
    model.addConstrs((complete_times[i2, h2, j2] >= complete_times[i1, h1, j1] + assignment[i2, h2, j2, k2] * process_time[(i2, j2, k2)]
                      - 30000 * (1 - order_on_machine[(i1, j1, i2, j2)]) - 30000 * (2 - assignment[i1, h1, j1, k1] - assignment[i2, h2, j2, k2])
                      for i1 in id_job for i2 in id_job for h1 in id_combination[i1] for h2 in id_combination[i2] for j1 in id_set_operation[i1][h1]
                      for j2 in id_set_operation[i2][h2] for k1 in machine_oper[(i1, j1)] for k2 in machine_oper[(i2, j2)] if (i1 != i2) and (k1 == k2)), name="ma_1")

    # Constraint 8
    model.addConstrs((complete_times[i1, h1, j1] >= complete_times[i2, h2, j2] + assignment[i1, h1, j1, k1] * process_time[(i1, j1, k1)]
                      - 30000 * order_on_machine[(i1, j1, i2, j2)] - 30000 * (
                          2 - assignment[i1, h1, j1, k1] - assignment[i2, h2, j2, k2])
                     for i1 in id_job for i2 in id_job for h1 in id_combination[i1] for h2 in id_combination[i2] for j1 in id_set_operation[i1][h1]
                     for j2 in id_set_operation[i2][h2] for k1 in machine_oper[(i1, j1)] for k2 in machine_oper[(i2, j2)] if (i1 != i2) and (k1 == k2)), name="ma_2")

    # Constraint 9: define makespan
    model.addConstrs((makespan >= complete_times[i, h, j] for i in id_job for h in id_combination[i]
                      for j in id_set_operation[i][h]), name="span")

    model.setObjective(makespan, GRB.MINIMIZE)

    # warm start
    if drl_sol:
        model.setParam("MIPStart", drl_sol)
    model.setParam(GRB.Param.Threads, 64)
    print('Solving model')

    model.optimize()


    status = model.status
    if status == GRB.OPTIMAL:
        model.write("sol.sol")
        print("Found an optimal solution.")
    elif status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("error.ilp")
        print("Model is infeasible.")
    elif status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    else:
        print("Unknown ", status)

    if status == GRB.OPTIMAL or status == GRB.FEASIBLE:
        for i in id_job:
            for h in id_combination[i]:
                for j in id_set_operation[i][h]:
                    for k in machine_oper[(i, j)]:
                        if assignment[i, h, j, k].x > 0.5: 
                            start = complete_times[i, h, j].x - process_time[(i, j, k)]
                            end = complete_times[i, h, j].x
                            assignment_value = assignment[i, h, j, k].x
                            print( f"Job {i} operation {j} start time on machine {k}: {round(start)}, end time: {round(end)}, allocation: {round(assignment_value)}")

        print("Makespan:", round(makespan.x))
        if save_path:
            if status == GRB.OPTIMAL:
                split_path = save_path.split('.')
                save_path = split_path[0] + '_optimal.' + split_path[1]
            schedule = sorted(schedule, key=lambda x: x[3])
            with open(save_path, 'w') as file:
                file.write(str(makespan.x) + '\n')
                for row in schedule:
                    file.write(' '.join(map(str, row)) + '\n')

        print("Total makespan:", round(makespan.x))
        
    else:
        print("No feasible solution found.")
        split_path = save_path.split('.')
        save_path = split_path[0] + '_infeasible.' + split_path[1]
        with open(save_path, 'w') as file:
            file.write("Infeasible")


if __name__ == '__main__':
    # File paths and directories
    file_folder = "problem_generate/0729/or_problem_job5_mas3"
    save_folder = "IL_test/0729/Gurobi/job5_mas3"
    ws_folder = "data_test/kim/ws_sol/"
    is_ws = True
    only_comb = False

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for file in os.listdir(file_folder):
        print(file)

        file_path = os.path.join(file_folder, file)
        save_path = os.path.join(save_folder, "o2d_sol_" + file)
        ws_path = os.path.join(ws_folder, "ws_sol_" + file) if is_ws else ''

        # Check if the solution file already exists, if so, skip this file.
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
            ope_or, ope_req_num, ope_eligible, _, _, _ = load_ipps(lines)

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

        solve_ipps_with_gurobi(id_job, id_machine, id_operation, id_set_operation, id_combination,
                               process_time, machine_oper, ope_ma_adj, matrix_pre_proc, matrix_cal_cumul,
                               drl_sol=ws_path if os.path.exists(ws_path) else None, save_path=save_path, only_comb=only_comb)
