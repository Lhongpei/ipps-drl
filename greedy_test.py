import copy
import json
import os
import random
import time as time

import pandas as pd
import torch
import numpy as np

import pynvml

from env.load_data import nums_detec

from greedy.greedy_rules import greedy_rule
import sys
from omegaconf import OmegaConf
from env.ipps_env import IPPSEnv
from utils.trick import shrink_schedule
from collections import defaultdict
sys.path.append('.')
sys.path.append('..')

from draw_gantt import draw_sol_gantt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    conf = OmegaConf.load("./config.yaml")
    env_paras = conf["env_paras"]
    env_paras.is_greedy = True
    model_paras = conf["nn_paras"]
    test_paras = conf["test_paras"]
    env_paras["device"] = device
    env_paras.batch_size = 1
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]

    data_path = "./data_test/{0}/problem/".format(test_paras["data_path"])
    num_ins = len(os.listdir(data_path))
    print("num_ins: ", num_ins)
    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[:num_ins]
    mod_files = os.listdir('./model/')[:]

    envs = []  # Store multiple environments

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    # Schedule instance by instance
    step_time_last = time.time()
    result = pd.DataFrame()
    result['files'] = test_files
    num_sample = 50
    for ope_rule in ['MWKR', 'MOR', 'FIFO', 'Muhammad']:
        for ma_rule in ['SPT', 'EFT', 'LUM']:
            times = []
            rule = ope_rule + '_' + ma_rule
            print(f"Using rule: {rule}")
            best_makespans = []
            mean_makespans = []
            for i_ins in range(num_ins):
                test_file = data_path + test_files[i_ins]
                with open(test_file) as file_object:
                    line = file_object.readlines() 
                    ins_num_jobs, ins_num_mas, _ = nums_detec(line)
                env_test_paras["num_jobs"] = ins_num_jobs
                env_test_paras["num_mas"] = ins_num_mas

                # Environment object already exists
                if len(envs) == num_ins:
                    env = envs[i_ins]
                # Create environment object
                else:
                    # Clear the existing environment

                    # DRL-S, each env contains multiple (=num_sample) copies of one instance
                    env = IPPSEnv(case=[test_file], env_paras=env_test_paras, data_source='file')
                    envs.append(copy.deepcopy(env))
                    print("Create env[{0}]".format(i_ins))

                # Schedule an instance/environment
                ms_mean = 0
                ms_best = np.inf
                for _ in range(num_sample):
                    makespan, time_re, _ = schedule(env, ope_rule, ma_rule)
                    ms_mean += makespan
                    ms_best = min(ms_best, makespan)
                ms_mean /= num_sample
                best_makespans.append(ms_best)
                mean_makespans.append(ms_mean)
                times.append(time_re)
                # DRL-G
                print("best makespan: ", ms_best)
                print("mean makespan: ", ms_mean)
                print("finish env {0}".format(i_ins))
            result[rule + '_best'] = best_makespans
            result[rule + '_mean'] = mean_makespans
            result[rule + '_time'] = np.mean(times)
            print("rule_spend_time: ", time.time() - step_time_last)
            print("best_makespans_avg: ", np.mean(best_makespans))

    print("total_spend_time: ", time.time() - start)
    result.to_csv(f"greedy/result/greedy_result_huge_add.csv", index = False)

def schedule(env:IPPSEnv, opes_rule, mas_rule):
    # Get state and completion signal
    env.reset()
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    machine_end_times = np.zeros(env.num_mas)
    job_end_times = {job: 0 for job in range(env.num_jobs)}
    operation_times = defaultdict(lambda: (0,0))
    machine_start_pointer = defaultdict(list)
    end_start_pointer = defaultdict(lambda: [0])
    i = 0
    if opes_rule == 'Muhammad':
        rand_num_id = state.combs_id_batch.squeeze(0)*torch.rand_like(state.combs_id_batch)
        choice = rand_num_id.max(dim=2)[1].view(-1)
        T_m_all = state.graph.data['combs'].x[choice, 0]
        T_s = max(T_m_all)
        C_all = torch.zeros_like(T_m_all)
        for i, T_m in enumerate(T_m_all):
            if T_m >= 0.95 * T_s:
                C_all[i] = 5
            elif T_m >= 0.85 * T_s and T_m < 0.95 * T_s:
                C_all[i] = 4
            elif T_m >= 0.70 * T_s and T_m < 0.85 * T_s:
                C_all[i] = 3
            elif T_m >= 0.50 * T_s and T_m < 0.70 * T_s:
                C_all[i] = 2
            else:
                C_all[i] = 1
        p = C_all/C_all.sum()

    sol = []
    matrix_proc_time = state.proc_times_batch[0].cpu().numpy()
    machine_load = np.zeros(env.num_mas)
    while not done:
        i += 1
        with torch.no_grad():
            other_info = machine_end_times if mas_rule == 'EFT' else machine_load if mas_rule == 'LUM' else None
            if opes_rule == 'Muhammad':
                actions = greedy_rule(state, opes_rule, mas_rule, env.device, other_info=other_info, p=p)
            else:
                actions = greedy_rule(state, opes_rule, mas_rule, env.device, other_info=other_info)


        step = [actions[0].item(), actions[1].item(), actions[2].item()]
        operation, machine, job = step
        proc_time = matrix_proc_time[operation][machine]
        if machine != -1 and proc_time != 0:
            machine_load[machine] += 1
            if len(machine_start_pointer[machine]) == 0:
                start_time = job_end_times[job]
                end_time = start_time + proc_time
                machine_start_pointer[machine].append(start_time)
                end_start_pointer[machine].append(end_time)
            else:
                start_time_np = np.array(machine_start_pointer[machine])
                end_time_np = np.array(end_start_pointer[machine][:-1])
                idle_interval = start_time_np - end_time_np
                assert (idle_interval>=0).all()
                com = (idle_interval >= proc_time) & (start_time_np >= job_end_times[job] + proc_time)
                if np.sum(com) == 0:
                    start_time = max(end_start_pointer[machine][-1], job_end_times[job])
                    end_time = start_time + proc_time
                    machine_start_pointer[machine].append(start_time)
                    end_start_pointer[machine].append(end_time)
                else:
                    index = np.argmax(com)
                    start_time = max(end_time_np[index], job_end_times[job])
                    end_time = start_time + proc_time
                    machine_start_pointer[machine].insert(index, start_time)
                    end_start_pointer[machine].insert(index + 1, end_time)

            machine_end_times[machine] = end_start_pointer[machine][-1]
            job_end_times[job] = end_time
            operation_times[operation] = (start_time, end_time)
        sol.append(step)
        state, _, dones, _ = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)

    # Verify the solution
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error!!!!!!!")
    makespan, new_sol = shrink_schedule(sol, env.num_jobs, env.num_mas, state.proc_times_batch[0].cpu().numpy(), return_sol = True)
    draw_sol_gantt(new_sol)
    return makespan, spend_time, sol


        
if __name__ == '__main__':
    main()


  