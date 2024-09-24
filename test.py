import copy
import os
import random
import time as time

import pandas as pd
import torch
import numpy as np

import pynvml
from env.ipps_env import IPPSEnv
from models.memory import MemoryRL
from models.ppo import PPO
from env.load_data import nums_detec
from utils.trick import shrink_schedule
from utils.sol_convert import sort_sol

import sys
sys.path.append('.')
sys.path.append('..')
from omegaconf import OmegaConf


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    setup_seed(0)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if torch.cuda.is_available():
        device = "cuda:0"
        use_cuda = True
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"
        use_cuda = False
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    conf = OmegaConf.load("./config.yaml")
    env_paras = conf["env_paras"]
    model_paras = conf["nn_paras"]
    train_paras = conf["train_paras"]
    test_paras = conf["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)

    if test_paras["sample"]: env_test_paras["batch_size"] = test_paras["num_sample"]
    else: env_test_paras["batch_size"] = 1

    data_path = "./data_test/{0}/problem/".format(test_paras["data_path"])
    solution_path = "./data_test/{0}/drl_solution/".format(test_paras["data_path"])
    lb_path = "./data_test/{0}/solution/".format(test_paras["data_path"])
    has_lb = os.path.exists(lb_path)
    test_files = os.listdir(data_path)
    num_ins = len(test_files) if getattr(test_paras, "num_ins", None) is None else test_paras["num_ins"]
    print("Find {0} instances".format(num_ins))
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[:num_ins]
    mod_files = os.listdir('./model/')
    if_save_sols = test_paras["if_save_sols"]

    memories = MemoryRL()
    model = PPO(model_paras, train_paras)
    model.policy_old.eval()
    model.policy.eval()
    envs = []  # Store multiple environments
    rules = []
    # Detect and add models to "rules"
    for _, _, fs in os.walk('./model/'):
        for f in fs:
            if f.endswith('.pt'):
                rules.append(f)


    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/test_{0}'.format(str_time)
    os.makedirs(save_path)
    


    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if use_cuda:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location = device)
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location = 'cpu')
            
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        ave_makespans = []
        best_makespans = []
        times = []
        sols = []
        lbs = []
        for i_ins in range(num_ins):
            
            test_file = os.path.join(data_path, test_files[i_ins])
            print('test_file:', test_file)
            if has_lb:
                lb_file = os.path.join(lb_path, 'o2d_sol_' + '_optimal.'.join(test_files[i_ins].split('.') + 'sol'))
                lb_file = os.path.join(lb_path, 'o2d_sol_' + test_files[i_ins] + 'sol') if not os.path.exists(lb_file) else lb_file
            with open(test_file, ) as file_object:
                line = file_object.read().splitlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            if has_lb:
                with open(lb_file, ) as lb_file_object:
                    lines = lb_file_object.read().splitlines()
                    lbs.append(int(lines[0]))
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total > 0.7:
                    envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance
                if test_paras["sample"]:
                    env = IPPSEnv(case=test_file,
                                   env_paras=env_test_paras, data_source='copy_file')
                # DRL-G, each env contains one instance
                else:
                    env = IPPSEnv(case=[test_file], env_paras=env_test_paras, data_source='file')
                envs.append(copy.deepcopy(env))
                print("Create env[{0}]".format(i_ins))

            # Schedule an instance/environment
            # DRL-S
            if test_paras["sample"]:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                best_sol = None
                best_makespan = float('inf')
                for j in range(test_paras["num_average"]):
                    makespan, time_re, sol = schedule(env, model, memories, flag_sample=test_paras["sample"],
                                                      save_sols = if_save_sols, shrink = True)
                    makespan_s.append(makespan)
                    if torch.min(makespan) < best_makespan:
                        best_makespan = torch.min(makespan).item()
                        best_sol = sol
                    time_s.append(time_re)

                ave_makespans.append(torch.mean(torch.cat(makespan_s, dim=0)))
                best_makespans.append(torch.min(torch.cat(makespan_s, dim=0)))
                print(f'{i_ins}th instance, best_makespan: {best_makespan}')
                times.append(torch.mean(torch.tensor(time_s)))
                sols.append(sol)
            # DRL-G
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                best_sol = None
                best_makespan = float('inf')
                for j in range(test_paras["num_average"]):
                    makespan, time_re, sol = schedule(env, model, memories,
                                                      save_sols = if_save_sols, shrink = True)
                    makespan_s.append(makespan)
                    if torch.min(makespan) < best_makespan:
                        best_makespan = torch.min(makespan).item()
                        best_sol = sol
                    time_s.append(time_re)
                    env.reset()
                ave_makespans.append(torch.mean(torch.cat(makespan_s, dim=0)))
                best_makespans.append(torch.min(torch.cat(makespan_s, dim=0)))
                times.append(torch.mean(torch.tensor(time_s)))
                sols.append(sol)
                print(f'{i_ins}th instance, best_makespan: {best_makespan}')

            if if_save_sols:
            # Save the solution in a .solipps file
                if not os.path.exists(solution_path):
                    os.makedirs(solution_path)
                formatted_data = []
                for row in best_sol:
                    formatted_row = f"{int(row[0]):d} {int(row[1]):d} {int(row[2]):d} {row[3]} {row[4]}"
                    formatted_data.append(formatted_row)
                with open(os.path.join(solution_path, 'drl_sol_' + test_files[i_ins] + 'sol'), "w") as file:
                    # Transpose the solution to separate operation, machines, and jobs
                    file.write(str(best_makespan) + "\n")
                    file.write("\n".join(formatted_data))
                print("finish env {0}".format(i_ins))
        
            
        print("rule_spend_time: ", time.time() - step_time_last)

        # Save makespan and time data to files

        data = pd.DataFrame({
            'file_name': test_files,
            'ave_makespan': [tensor.item() for tensor in ave_makespans],
            'best_makespan': [tensor.item() for tensor in best_makespans],
            'ave_time': [tensor.item() for tensor in times],
            'lower_bound': lbs
        }) if has_lb else pd.DataFrame({
            'file_name': test_files,
            'ave_makespan': [tensor.item() for tensor in ave_makespans],
            'best_makespan': [tensor.item() for tensor in best_makespans],
            'ave_time': [tensor.item() for tensor in times]
        })

        data.to_csv(f'{save_path}/result_{str_time}_{rule.split(".")[0]}.csv', index=False)
        # np._save('{0}/sols_{1}.npy'.format(save_path, str_time), sols)

        for env in envs:
            env.reset()

    print("total_spend_time: ", time.time() - start)


def schedule(env:IPPSEnv, model, memories, flag_sample=False, save_sols = False, shrink = False):
    # Get state and completion signal
    state = env.reset()
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    i = 0
    steps_act = [[] for _ in range(state.batch_idxes.size(0))]
    best_sol = None
    for i in range(state.batch_idxes.size(0)):
        for j in range(env.num_ope_biases_batch[i].size(0)):
            steps_act[state.batch_idxes[i]].append([env.num_ope_biases_batch[i, j].item(), 0, 0])
            
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, flag_sample=flag_sample, flag_train=False)

            for i in range(state.batch_idxes.size(0)):
                steps_act[state.batch_idxes[i].item()].append(actions[:,i].cpu().tolist())
        
        state, _, dones, _ = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)

    makespan_batch = copy.deepcopy(env.makespan_batch)
    if shrink:
        sols = []
        for i in range(env.batch_size):
            makespan, sol = shrink_schedule(steps_act[i], env.num_jobs, env.num_mas, state.proc_times_batch[i].cpu().numpy(),
                                            ignore_supernode = True, return_sol = True)
            makespan_batch[i] = makespan
            sols.append(sol)
        print(env.makespan_batch - makespan_batch)
    # Verify the solution
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error!!!!")
    if save_sols:
        best_idx = torch.argmin(makespan_batch)
        best_sol = np.array(sort_sol(sols[best_idx], env.ope_pre_adj_batch[0])[1:]) if shrink else env.get_schedule(best_idx)
                
    return makespan_batch, spend_time, best_sol



if __name__ == '__main__':
    main()


  