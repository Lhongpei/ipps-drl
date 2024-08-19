
from env.ipps_env import IPPSEnv
from models.memory import MemoryRL, MemoryIL
import torch
import time
import os
import copy
from tqdm import tqdm
from utils.draw_gantt import draw_sol_gantt
def get_validate_env(env_paras, valid_data_files = None):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    if valid_data_files == None:
        file_path = "./data_dev/{0}{1}/".format(str.zfill(str(env_paras["num_jobs"]),2), str.zfill(str(env_paras["num_mas"]),2))
        valid_data_files = os.listdir(file_path)
        for i in range(env_paras['batch_size']):
            valid_data_files[i] = file_path+valid_data_files[i]
    env = IPPSEnv(case=valid_data_files[:env_paras['batch_size']], env_paras=env_paras, data_source='file')
    print('There are {0} dev instances in Validate datasets.'.format(env_paras["batch_size"]))  # validation set is also called development set
    return env

def validate(env_paras, env, model_policy, DRL = True, draw = False, gantt_path = None, N = None):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = MemoryRL() if DRL else MemoryIL()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.reset()
    
    done = False
    dones = env.done_batch
    model_policy.eval()
    while ~done:
        with torch.no_grad():
            if DRL:
                actions = model_policy.act(state, memory, flag_sample = False, flag_train = False)
            else:
                 actions = model_policy.act(state, memory, flag_sample = False)
        state, rewards, dones, info = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error!!!!!!!")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    if draw:
        schedule = env.get_schedule(0)
        formatted_data = []
        for row in schedule:
            formatted_row = f"{int(row[0]):d} {int(row[1]):d} {int(row[2]):d} {row[3]} {row[4]}"
            formatted_data.append(formatted_row)
        sol = [str(env.makespan_batch[0].item())] + formatted_data
        draw_sol_gantt(sol, folder = gantt_path, suffix = N)


    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch
