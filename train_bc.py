import torch
from omegaconf import OmegaConf
import os
from env.ipps_env import IPPSEnv
from utils.utils import solutions_padding
from validate import validate, get_validate_env
from models.bc import BehaviorCloning
from dataset import ILDataScheduler, collate_fn
from models.memory import MemoryIL
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
from collections import deque
import wandb
import time
def init_wandb(use_wandb):
    if use_wandb:
        wandb.init(
            group='IL',
            name = '0505 bc all qkv',
            # set the wandb project where this run will be logged
            project="PyG implementation",
            entity="ipps-learning"
            # track hyperparameters and run metadata
        )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':

    use_wandb = True
    epoch = 30000
    init_wandb(use_wandb)
    setup_seed(42)
    
    if torch.cuda.is_available():
        device = "cuda:3"
        torch.cuda.set_device(device)
        torch.set_default_device(device)
    else:
        device = "cpu"
        torch.set_default_device(device)
    
    config = OmegaConf.load('config.yaml')
    config.nn_paras.device = device
    config.env_paras.device = device
    env_paras = config['env_paras']
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    IL_paras = config['IL_paras']
    config_dict = OmegaConf.to_container(config, resolve=True)
    if use_wandb:
        wandb.config.update(config_dict)
    dir_dict = "IL_test/{0}{1}/".format(str.zfill(str(env_paras["num_jobs"]),2),
                                                   str.zfill(str(env_paras["num_mas"]),2))

    dir_dict = "IL_test/0505_all/"
    scheduler = ILDataScheduler(config, dir_dict, device)
    dataset_loader = scheduler.load_dataset(shuffle=False)
    validate_path = "./data_dev/{0}{1}/".format(str.zfill(str(env_paras["num_jobs"]),2),
                                            str.zfill(str(env_paras["num_mas"]),2))
    
    validate_path = dir_dict + "problem/"
    is_validate = os.path.exists(validate_path)
    valid_data_files = None

    valid_data_files = os.listdir(validate_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = validate_path+valid_data_files[i]


    if is_validate:
        print("Validation data exists")
        env_valid = get_validate_env(env_valid_paras, valid_data_files)  # Create an environment for validation
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_il_{0}'.format(str_time)
    os.mkdir(save_path)
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')
    model = BehaviorCloning(config)

    # pretrain_model_path = os.path.join('./model', os.listdir('./model')[0])
    # model.expert.policy.load_state_dict(torch.load(pretrain_model_path))

    memories = MemoryIL()
    print("start training...")
    i = 0
    max_batch_size = env_paras["batch_size"]
    for ep in range(epoch):
        dataset_loader = scheduler.load_dataset(shuffle=False)
        for dataset in dataset_loader:
            for problems, solutions in DataLoader(dataset,
                                                  batch_size = max_batch_size, 
                                                  shuffle = True, 
                                                  collate_fn=collate_fn,
                                                  generator=torch.Generator(device='cuda')):
                i += 1
                env_paras["batch_size"] = len(solutions)
                env = IPPSEnv(case = problems, env_paras=env_paras, data_source='tensor')

                state = env.reset()
                start_state = env.num_ope_biases_batch.tolist()
                solutions = solutions_padding(solutions)
                model.give_solution(solutions)
                done = False
                dones = env.done_batch
                tot_loss = []
                while ~done:
                    with torch.no_grad():
                        actions = model.expert.act_expert(state, memories)
                    assert actions[0].size(0) == state.batch_idxes.size(0)
                    # print(actions)
                    state, rewards, dones, info = env.step(actions)
                    done = dones.all()
                if i % IL_paras.update_timestep == 0:
                    loss = model.update(memories)
                    print(f'EPOCH {i}, Loss: {loss}')
                    if use_wandb:
                        wandb.log({"training_loss": loss}, step=i)
                        wandb.log({'gpu_usage': torch.cuda.memory_allocated()} , step=i)
                        wandb.log({'gpu_reserved': torch.cuda.memory_reserved()} , step=i)
                    tot_loss.append(loss)
                    memories = MemoryIL()

                if i % IL_paras["save_timestep"] == 0:
                    torch.cuda.empty_cache()
                    if is_validate:
                        print('\nStart validating')
                        vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.expert, DRL=False,
                                                                draw = True, gantt_path = save_path, N = i)
                        # Save the best model if new best is found
                        if vali_result < makespan_best:
                            makespan_best = vali_result
                            if len(best_models) == maxlen:
                                delete_file = best_models.popleft()
                                os.remove(delete_file)
                            save_file = '{0}/save_best_IL_{1}.pt'.format(save_path, i)
                            best_models.append(save_file)
                            torch.save(model.expert.policy.state_dict(), save_file)
                        
                        elif i % 10*IL_paras["save_timestep"] == 0:
                            save_file = '{0}/save_IL_{1}.pt'.format(save_path, i)
                            torch.save(model.expert.policy.state_dict(), save_file)
                            
                        elif torch.abs(vali_result - makespan_best) < 1:
                            save_file = '{0}/save_sub_best_IL_{1}.pt'.format(save_path, i)
                            torch.save(model.expert.policy.state_dict(), save_file)
                            
                        print(f'EPOCH {i}, Validation makespan: {vali_result}, Best makespan: {makespan_best}')
                        
                        if use_wandb:
                        # Log validation metrics to wandb
                            wandb.log({"validation_makespan": vali_result.item()}, step=i)
                    else:
                        save_file = '{0}/save_IL_{1}.pt'.format(save_path, i)
                        torch.save(model.expert.policy.state_dict(), save_file)
