import copy
import os
import random
import time as time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import torch
import numpy as np
import pickle
import pynvml
from env.ipps_env import IPPSEnv
from models.memory import MemoryRL
from models.ppo import PPO
from env.load_data import nums_detec, load_ipps
from env.load_fjsp import nums_detec_fjsp
from utils.trick import shrink_schedule
from sol_convert import sort_sol
from utils.mcts_v3 import mcts,tree_parallel,leaf_parallel
import utils.IPPS_ENV_CPP_v2.pywrap.env_wrapper as env_wrap
import re
import sys
sys.path.append('.')
sys.path.append('..')
from omegaconf import OmegaConf
from draw_gantt import draw_gantt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    results=[]
    action_dict = {}  # 用于保存每个 problem 对应的调度轨迹
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    setup_seed(0)
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if torch.cuda.is_available():
        device = "cuda:2"
        use_cuda = True
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"
        use_cuda = False
        torch.set_default_tensor_type('torch.FloatTensor')
    # device="cpu"
    # use_cuda = True
    print("PyTorch device: ", device)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    # with open("./config.json", 'r') as load_f:
    #     conf = json.load(load_f)
    conf = OmegaConf.load("./config.yaml")
    env_paras = conf["env_paras"]
    model_paras = conf["nn_paras"]
    train_paras = conf["train_paras"]
    test_paras = conf["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    
    #num_ins = test_paras["num_ins"]
    if test_paras["sample"]: env_test_paras["batch_size"] = test_paras["num_sample"]
    else: env_test_paras["batch_size"] = 1

    # for d_path in ["0405", "0503", "0605", "1620", "2025"]:
    for d_path in ["0605"]:
        test_paras["data_path"] = d_path
        # data_path = "./data_test/{0}/problem/".format(test_paras["data_path"])
        # solution_path = "./data_test/{0}/drl_solution/".format(test_paras["data_path"])
        # lb_path = "./data_test/{0}/solution/".format(test_paras["data_path"])
        data_path = "./data_test/{0}/problem/".format(test_paras["data_path"])
        solution_path = "./data_test/{0}/drl_solution/".format(test_paras["data_path"])
        lb_path = "./data_test/{0}/solution/".format(test_paras["data_path"])
        has_lb = False # os.path.exists(lb_path)
        test_files = os.listdir(data_path)
        # num_ins = len(test_files)
        # print("Find {0} instances".format(num_ins))
        test_files.sort(key=lambda x: x[:-4])
        num_ins=len(test_files)
        test_files = test_files[:num_ins]
        mod_files = os.listdir('./model/')

        # data_path = './'
        # test_files = ["data_test/dumb/problem/dumb.txt"]

        memories = MemoryRL()
        model = PPO(model_paras, train_paras)
        model.policy_old.eval()
        model.policy.eval()
        rules = test_paras["rules"]
        envs = []  # Store multiple environments

        # Detect and add models to "rules"
        if "DRL" in rules:
            for root, ds, fs in os.walk('./model/'):
                for f in fs:
                    if f.endswith('.pt'):
                        rules.append(f)
        if len(rules) != 1:
            if "DRL" in rules:
                rules.remove("DRL")

        # Generate data files and fill in the header
        save=True
        if save:
            # str_time = "fivegreedy"
            str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
            save_path = f'./save/test_{test_paras["data_path"]}_{str_time}'
            os.makedirs(save_path,exist_ok=True)
        


        # Rule-by-rule (model-by-model) testing
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
            mcts_agent = mcts(model, timeLimit=1200, explorationConstant=5)
            # Schedule instance by instance

            lbs = []
            for i_ins in range(num_ins):
                
                test_file = os.path.join(data_path, test_files[i_ins])
                print('test_file:', test_file)
                lb_dict = {
                    1: 427, 2: 343, 3: 344, 4: 306, 5: 318, 6: 427,
                    7: 372, 8: 343, 9: 427, 10: 427, 11: 344, 12: 318,
                    13: 427, 14: 372, 15: 427, 16: 427, 17: 344, 18: 318,
                    19: 427, 20: 372, 21: 427, 22: 427, 23: 372, 24: 427
                }
                lb=-1
                if "kim" in test_file:
                    match = re.search(r'problem(\d+)', test_file)
                    if match:
                        problem_id = int(match.group(1))  # 提取为整数 1
                        print(problem_id)
                    lb=lb_dict.get(problem_id, -1)  
                    print("the lower bound is:", lb)

                if has_lb:
                    lb_file = os.path.join(lb_path, 'o2d_sol_' + '_optimal.'.join(test_files[i_ins].split('.')))
                    lb_file = os.path.join(lb_path, 'o2d_sol_' + test_files[i_ins]) if not os.path.exists(lb_file) else lb_file
                is_ipps = test_file.split('.')[-1] != 'fjs'
                lines = env_wrap.read_lines(test_file)
                
                start_nodes=[]
                for line in lines:
                    if line.endswith('start'):
                        try:
                            print(line)
                            start = int(line.split()[0])
                            print(start)

                            start_nodes.append(start)
                        except (ValueError, IndexError):
                            print(f"处理以下行时出错: {line}")

                ins_num_jobs, ins_num_mas, _ = nums_detec(lines) if is_ipps else nums_detec_fjsp(line)
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
                parallel_num=4
                # CyEnv_list=[env_wrap.PyEnv(lines, is_eval=True)for _ in range(parallel_num)]
                # Schedule an instance/environment
                # time0 = time.time()
                starttime = time.time()
                CyEnv=env_wrap.PyEnv(lines, is_eval=True)
                
                print("kkkkkkkkkk")
                # CyEnv.printDebugInfo()
                # CyEnv.printDebugInfo()
                # CyEnv.printDebugInfo()
                print("88888888888888888888888")

                # fuck=CyEnv.copy()
                # print("wonima????")
                #action_list, best_makespan=leaf_parallel(env,CyEnv_list,start_nodes,model,lb,num=parallel_num,iterationLimit = 10000, explorationConstant=5)
                action_list, best_makespan = mcts_agent.search(env,CyEnv,start_nodes,lb)
                print(action_list)
                #action_list, best_makespan = tree_parallel(env,CyEnv_list,start_nodes,model,lb,parallel_num,iterationLimit=2000,explorationConstant=5)
                endtime = time.time()
                test_time = endtime - starttime
                # print("time: ", time.time() - time0)
                print("best-makespan is:",best_makespan)
                if save:
                    summary_path = os.path.join(save_path, "result_summary.csv")
                    if not os.path.exists(summary_path):
                        pd.DataFrame(columns=["problem", "model", "makespan", "test time"]).to_csv(summary_path, index=False)
                results.append({
                    "problem": test_files[i_ins],
                    "model": mod_files[i_rules] if rule.endswith('.pt') else rule,
                    "makespan": best_makespan,
                    "test time":test_time
                })
                result_row = pd.DataFrame([{
                    "problem": test_files[i_ins],
                    "model": mod_files[i_rules] if rule.endswith('.pt') else rule,
                    "makespan": best_makespan,
                    "test time": test_time
                }])
                if save:
                    result_row.to_csv(summary_path, mode='a', header=False, index=False)
                    problem_name = test_files[i_ins]
                    model_name = mod_files[i_rules] if rule.endswith('.pt') else rule
                    key = f"{problem_name}__{model_name}"
                    action_dict[key] = action_list   
                    df = pd.DataFrame({
                        "best_makespan": mcts_agent.bestmakespan_list,
                        "round_makespan": mcts_agent.makespan_list
                    })

                    # 保存到CSV文件
                    df.to_csv(os.path.join(save_path,f"{problem_name}__{model_name}mcts_search_record.csv"), index_label="iteration", encoding="utf-8-sig")


        if save:
            df_result = pd.DataFrame(results)
            df_result.to_csv(os.path.join(save_path, "result_summary.csv"), index=False)           
                    # makespan, time_re, sol = schedule(env, model, memories, flag_sample=test_paras["sample"],
            #                                     save_sols = if_save_sols, draw = if_draw_gantt, shrink = True)
            with open(os.path.join(save_path, "action_dict.pkl"), "wb") as f:
                pickle.dump(action_dict, f)


def schedule(env:IPPSEnv, model, memories, flag_sample=False, draw = True, save_sols = False, shrink = False):
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
        
        state, rewards, dones,info = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    # print("spend_time: ", spend_time)
    # print("spend_time: ", spend_time)
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