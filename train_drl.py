import copy
import os
import random
import time
from collections import deque
from env.ipps_env import IPPSEnv
import pandas as pd
import torch
import numpy as np
import wandb
from models.memory import MemoryRL
from models.ppo import PPO
from omegaconf import OmegaConf
from generator.case_generator_ipps import CaseGenerator
from validate import validate, get_validate_env

import tqdm
def init_wandb(use_wandb):
    if use_wandb:
        wandb.init(
            group='DRL',
            name = '0503 naive 0.99',
            project="PyG implementation",
            entity="ipps-learning"
        )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    use_wandb = False
    init_wandb(use_wandb)
    setup_seed(0)
    # PyTorch initialization

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
        torch.set_default_device(device)
    else:
        device = "cpu"
        torch.set_default_device(device)
    print("PyTorch device: ", device)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    conf = OmegaConf.load("./config.yaml")
    
    config_dict = OmegaConf.to_container(conf, resolve=True)
    if use_wandb:
        wandb.config.update(config_dict)
        
    env_paras = conf.env_paras
    model_paras = conf.nn_paras
    train_paras = conf.train_paras
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    
    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    
    memories = MemoryRL()
    model = PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])

    # pretrain_model_path = os.path.join('./model', os.listdir('./model')[0])
    # # model.load_pretrained_policy(pretrain_model_path)\
    # model.policy.load_state_dict(torch.load(pretrain_model_path))
    # model.policy_old.load_state_dict(torch.load(pretrain_model_path))

    is_validate = os.path.exists("./data_dev/{0}{1}/".format(str.zfill(str(env_paras["num_jobs"]),2),
                                                  str.zfill(str(env_paras["num_mas"]),2)))
    is_validate = True
    if is_validate:
        env_valid = get_validate_env(env_valid_paras)
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)
     
    # Training curve storage path (average of validation set)
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    # Training curve storage path (value of each validating instance)
    writer_100 = pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time))
    valid_results = []
    valid_results_100 = []
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)
    writer_ave._save()
    writer_ave.close()
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_100, sheet_name='Sheet1', index=False)
    writer_100._save()
    writer_100.close()
    # addcase = CaseGenerator(1, num_mas, job_folder='./env/job_with_mas_5/')
    # Start training iteration
    start_time = time.time()
    env = None

    print("Start training...")
    for i in tqdm.tqdm(range(1, train_paras["max_iterations"]+1), desc="Training"):
        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            if env != None:
                env.close()
            # \mathcal{B} instances use consistent operations to speed up training
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, job_folder=f'./env/job_with_mas_{num_mas}/')
            env = IPPSEnv(case=case, env_paras=env_paras)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.reset()
        #state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, flag_train=True)
            assert actions[0].size(0) == state.batch_idxes.size(0)
            state, rewards, dones, info = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time()-last_time)
        
        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if use_wandb:
            wandb.log({'train_makespan':env.makespan_batch.mean()}, step=i)
            # wandb.log({'gpu_usage': torch.cuda.memory_allocated()} , step=i)
            # wandb.log({'gpu_reserved': torch.cuda.memory_reserved()} , step=i)
        if not gantt_result:
            raise ValueError("Scheduling Error!!!!!!!")
        # print("Scheduling Finish")
        env.reset()
        print('Updating')
        if i % train_paras["update_timestep"] == 0:
            loss_dict, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss_dict['total_loss'])
            memories.clear_memory()

            if use_wandb:
                # Log training metrics to wandb
                wandb.log({"reward": reward}, step=i)
                wandb.log(loss_dict, step=i)

            if i % train_paras["save_timestep"] == 0:
                torch.cuda.empty_cache()
                if is_validate:
                    print('\nStart validating')
                    vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old,
                                                            draw = True, gantt_path = save_path, N = i)
                    valid_results.append(vali_result.item())
                    valid_results_100.append(vali_result_100)
                    # Save the best model if new best is found
                    if vali_result < makespan_best:
                        makespan_best = vali_result
                        if len(best_models) == maxlen:
                            delete_file = best_models.popleft()
                            os.remove(delete_file)
                        save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                        best_models.append(save_file)
                        torch.save(model.policy.state_dict(), save_file)
                    
                    elif i % 10*train_paras["save_timestep"] == 0:
                        save_file = '{0}/save_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                        torch.save(model.policy.state_dict(), save_file)
                        
                    elif torch.abs(vali_result - makespan_best) < 1:
                        save_file = '{0}/save_sub_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                        torch.save(model.policy.state_dict(), save_file)
                    
                    if use_wandb:
                    # Log validation metrics to wandb
                        wandb.log({"validation_makespan": vali_result.item()}, step=i)
                        
                    print(f'EPOCH {i}, Validation makespan: {vali_result}, Best makespan: {makespan_best}')
                else:
                    torch.save(model.policy.state_dict(), '{0}/save_{1}_{2}_{3}_train_end.pt'.format(save_path, num_jobs, num_mas, i))


    # Save the data of training curve to files
    if is_validate:
        data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
        data.to_excel(writer_ave, sheet_name='Sheet1', index=False, startcol=1)
        writer_ave._save()
        writer_ave.close()
        column = [i_col for i_col in range(100)]
        data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
        data.to_excel(writer_100, sheet_name='Sheet1', index=False, startcol=1)
        writer_100._save()
        writer_100.close()
    if use_wandb:
            wandb.finish(  )
    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()