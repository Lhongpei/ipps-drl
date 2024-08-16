import torch
from dataclasses import dataclass
from env.load_data import load_ipps, nums_detec
from env.load_fjsp import load_fjsp, nums_detec_fjsp
import numpy as np
import copy
import sys
from utils.utils import pad_1d_tensors, pad_2d_tensors, pad_to_given_size, pad_to_middle_given_size, pad_stack_add_idxes, getAncestors, sort_schedule
from tqdm import tqdm
from network.hetero_data import Graph_Batch
from env.case_generator_ipps import CaseGenerator

@dataclass
class EnvState:
    """
    Represents the state of the environment in the IPPS simulation.
    """
    batch_idxes: torch.Tensor = None
    graph: Graph_Batch = None
    proc_times_batch: torch.Tensor = None
    opes_appertain_batch: torch.Tensor = None
    eligible_pairs: torch.Tensor = None
    future_eligible_pairs: torch.Tensor = None
    
    #Used in hand-craft PDR
    combs_id_batch: torch.Tensor = None
    combs_batch: torch.Tensor = None
    combs_time_batch: torch.Tensor = None
    remain_opes_batch: torch.Tensor = None
    ready_time: torch.Tensor = None
    next_ope_eligible_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None

    def update(self, batch_idxes, graph, proc_times_batch, opes_appertain_batch,
        eligible_pairs, future_eligible_pairs, combs_id_batch = None, combs_batch = None,
        combs_time_batch = None, remain_opes_batch = None, ready_time = None, next_ope_eligible_batch = None ,
        ope_ma_adj_batch = None):
        """
        Updates the state of the environment.
        """
        self.batch_idxes = batch_idxes
        self.graph = graph
        self.proc_times_batch = proc_times_batch
        self.opes_appertain_batch = opes_appertain_batch
        self.eligible_pairs = eligible_pairs
        self.future_eligible_pairs = future_eligible_pairs
        
        #Used in hand-craft PDR
        self.combs_id_batch = combs_id_batch
        self.combs_batch = combs_batch
        self.combs_time_batch = combs_time_batch
        self.remain_opes_batch = remain_opes_batch
        self.ready_time = ready_time
        self.next_ope_eligible_batch = next_ope_eligible_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch
        
def convert_info_job_2_ope(info_job_batch, opes_appertain_batch):
    '''
    Convert job infoures into operation infoures (such as dimension)
    '''
    return info_job_batch.gather(1, opes_appertain_batch)

class IPPSEnv:
    '''
    IPPS environment
    '''
    def __init__(self, case, env_paras, data_source = 'case'):
        '''
        Parameters
        - case: function or list or dict
            - function: case generator
            - list: list of file paths
            - dict: 
                - tensor: list of tensors
                - info: list of instance basic information (num_jobs, num_mas, num_opes)
        - env_paras: dict
        - data_source: str
            - 'case': generate instances through case generator
            - 'file': load instances from files
            - 'tensor': load instances from tensors
            - 'copy_file': copy a instance multiple times to form a batch
        '''
        #space
        # load paras
        # static
        self.batch_size = env_paras['batch_size']
        self.device = env_paras['device']
        self.reward_info = env_paras['reward_info']
        self.is_greedy = env_paras['is_greedy']
        self.change_interval = env_paras['proc_time_change_interval'] if env_paras['proc_time_change_interval'] > 0 else sys.maxsize
        self.N = 0
        # graph
        self.graph = Graph_Batch()
        # load instance
        num_data = 13 # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        self.num_mas = 0
        self.num_jobs = 0
        self.is_add_job = False
        self.valid_opes_num = torch.zeros(self.batch_size).long()
        self.valid_mas_num = torch.zeros(self.batch_size).long()
        self.valid_combs_num = torch.zeros(self.batch_size).long()
        self.valid_jobs_num = torch.zeros(self.batch_size).long()
        lines = []
        if data_source == 'case':  # Generate instances through generators
            for i in range(self.batch_size):
                lines.append(case.get_case())  # Generate an instance and save it
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
                self.num_mas = max(self.num_mas, num_mas)
                self.num_jobs = max(self.num_jobs, num_jobs)
        
        elif data_source=='tensor':  # Load instances from tensors
            tensor_loaded = case['tensor']
            infos = case['info']
            self.num_opes = max([info[2] for info in infos])
            self.num_mas = infos[0][1]
            self.num_jobs = infos[0][0]
        
        elif data_source=='file':
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.read().splitlines()
                    lines.append(line)
                num_jobs, num_mas, num_opes = nums_detec(lines[i]) 
                self.num_opes = max(self.num_opes, num_opes)
                self.num_mas = max(self.num_mas, num_mas)
                self.num_jobs = max(self.num_jobs, num_jobs)
                
        elif data_source=='copy_file':
            with open(case) as file_object:
                line = file_object.read().splitlines()
                num_jobs, num_mas, num_opes = nums_detec(line) 
                self.num_opes = num_opes
                self.num_mas = num_mas
                self.num_jobs = num_jobs
        
        self.ori_num_opes = self.num_opes
        self.ori_num_mas = self.num_mas
        self.ori_num_jobs = self.num_jobs 
        # load feats and paths
        graph_list = []

        if data_source != 'copy_file':
            for i in tqdm(range(self.batch_size),desc="Loading graph features and paths"):
                
                if data_source=='tensor':
                    load_data = tensor_loaded[i]
                else:
                    load_data = load_ipps(lines[i])

                load_graph = self.graph.load_features(load_data)
                self.valid_opes_num[i] = load_graph['opes'].x.size(0)
                self.valid_mas_num[i] = load_graph['mas'].x.size(0)
                self.valid_combs_num[i] = load_graph['combs'].x.size(0)
                self.valid_jobs_num[i] = load_graph['jobs'].x.size(0)
                graph_list.append(load_graph)
                for j in range(num_data):
                    tensors[j].append(load_data[j])
        else:
            print("Copying instances")
            load_data = load_ipps(line)
            load_graph = self.graph.load_features(load_data)
            graph_list = [copy.deepcopy(load_graph) for _ in range(self.batch_size)]
            tensors = [[copy.deepcopy(load_data[j]) for _ in range(self.batch_size)] for j in range(num_data)]

        # dynamic feats
        # shape: (batch_size, num_jobs, num_combs)
        self.combs_id_batch = pad_2d_tensors(tensors[10], value = 0, pad_dim = 2)
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = pad_2d_tensors(tensors[0], value=0, pad_dim=1).float()
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = pad_2d_tensors(tensors[1], value=0, pad_dim=1).long()
        # shape: (batch_size, num_opes)
        self.remain_opes_batch = pad_1d_tensors(tensors[12], value = 0).long()
        
        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = pad_2d_tensors(tensors[2], value=0, pad_dim=2).long()
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = pad_1d_tensors(tensors[3], value=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = pad_1d_tensors(tensors[4], value=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = pad_1d_tensors(tensors[5], value=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size, num_opes, num_opes), whether exist OR between operations
        self.ope_or_batch = pad_2d_tensors(tensors[7], value=0, pad_dim=2).long()
        # shape: (batch_size, num_combs, num_opes)
        self.combs_batch = pad_2d_tensors(tensors[11], value = 0, pad_dim = 2).float()


        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        # shape: (batch_size, num_opes) Whether the operation is eligible
        self.next_ope_eligible_batch = pad_1d_tensors(tensors[9], value=0).long()
        # shape: (batch_size, num_opes), the number of required operations for each operation
        self.ope_req_num_batch = pad_1d_tensors(tensors[8], value=100).long()
        # shape: (batch_size, num_jobs), the completion time of each job
        self.job_end_time_batch = torch.zeros(size=(self.batch_size, self.num_jobs))
        # shape: (batch_size, num_opes), the ready time of each operation
        self.ready_time = torch.zeros_like(self.ope_req_num_batch)
        # shape: (batch_size, num_combs) 
        self.combs_time_batch = torch.zeros(size=(self.batch_size, self.combs_batch.size(1)))
        self.job_estimate_end_batch = torch.zeros(size=(self.batch_size, self.num_jobs))
       
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # operation infoures        size = (self.batch_size, self.num_opes)
        # info_opes_batch[:, 0, :]
        self.info_ope_status_batch = torch.zeros(size = (self.batch_size, self.num_opes)).scatter(1, self.num_ope_biases_batch, 1)
        # info_opes_batch[:, 2, :]
        self.info_ope_proc_time_batch = torch.sum(self.proc_times_batch, dim=2).div(torch.count_nonzero(self.ope_ma_adj_batch, dim=2).float() + 1e-9) \
            if self.reward_info['ma_mean'] else torch.min(torch.where(self.ope_ma_adj_batch == 1, self.proc_times_batch, torch.inf), dim = 2)[0]
        self.info_ope_proc_time_batch = torch.where(self.info_ope_proc_time_batch == torch.inf, 0, self.info_ope_proc_time_batch)

        self.ave_proc_time = torch.sum(self.info_ope_proc_time_batch).div(torch.count_nonzero(self.info_ope_proc_time_batch).float() + 1e-9)

        self.info_ope_scheduled_start_batch = torch.zeros(size = (self.batch_size, self.num_opes))
        # info_ope_status_batch, info_ope_proc_time_batch, info_ope_scheduled_start_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))
        
        self.makespan_batch = self.get_estimate_end_time(self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.graph.init_from_graph_list(graph_list, self.combs_time_batch, self.remain_opes_batch, self.job_estimate_end_batch)

        self.init_state()
        
        self.backup4reset()
    
    
    def backup4reset(self):
        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)

        # info_opes_batch
        self.old_info_ope_status_batch = copy.deepcopy(self.info_ope_status_batch)
        self.old_info_ope_proc_time_batch = copy.deepcopy(self.info_ope_proc_time_batch)
        self.old_info_ope_scheduled_start_batch = copy.deepcopy(self.info_ope_scheduled_start_batch)
        self.old_remain_opes_batch = copy.deepcopy(self.remain_opes_batch)
        
        self.old_state = copy.deepcopy(self.state)
        self.old_graph = copy.deepcopy(self.graph)
        self.old_edge_indices = copy.deepcopy(self.graph.data._slice_dict)
        self.old_ope_req_num_batch = copy.deepcopy(self.ope_req_num_batch)
        self.old_next_ope_eligible_batch = copy.deepcopy(self.next_ope_eligible_batch)


        # combination info
        self.old_combs_id_batch = copy.deepcopy(self.combs_id_batch)
        self.old_comb_time_batch = copy.deepcopy(self.combs_time_batch)
    
    def update_graph(self, actions):
        self.graph.update_features(self.batch_idxes, actions, self.schedules_batch[:,:,3], self.ready_time, self.time,
                                self.machines_batch, self.ope_req_num_batch, self.combs_time_batch, 
                                (~self.mask_job_finish_batch).unsqueeze(-1)*self.combs_id_batch, self.remain_opes_batch, self.job_estimate_end_batch)

    def init_state(self):
        '''
        Initialize the state of the environment
        '''
        eligible_pairs, future_eligible_pairs = self.find_eligible_pairs(find_future=True)
        if self.is_greedy:
            self.state = EnvState(
                                    self.batch_idxes, self.graph, self.proc_times_batch, self.opes_appertain_batch,
                                    eligible_pairs, future_eligible_pairs, self.combs_id_batch, self.combs_batch, 
                                    self.combs_time_batch, self.remain_opes_batch, self.ready_time, self.find_next_ope_eligible(),
                                    self.ope_ma_adj_batch
                                )
        else:
            self.state = EnvState(
                                    self.batch_idxes, self.graph, self.proc_times_batch, self.opes_appertain_batch,
                                    eligible_pairs, future_eligible_pairs
                                )    
    def update_state(self):
        eligible_pairs, future_eligible_pairs = self.find_eligible_pairs(find_future=True)
        if self.is_greedy:
            self.state.update(
                self.batch_idxes, self.graph, self.proc_times_batch, self.opes_appertain_batch,
                eligible_pairs, future_eligible_pairs, self.combs_id_batch, self.combs_batch, 
                self.combs_time_batch, self.remain_opes_batch, self.ready_time, self.find_next_ope_eligible(),
                self.ope_ma_adj_batch
            )
        else:
            self.state.update(
                self.batch_idxes, self.graph, self.proc_times_batch, self.opes_appertain_batch,
                eligible_pairs, future_eligible_pairs
            )
    def step(self, actions):
        '''
        Environment transition function
        '''

        # print(actions)
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]

        self.N += 1

        # Current processing instance in the batch
        batch_idxes = self.batch_idxes
        # Judge whether we will choose an O-M pair according to the actions(if action is -1 then we will not choose any pair)
        active_idxes = batch_idxes[opes != -1]
        wait_idxes = torch.where(opes == -1)[0]

        opes = opes[opes != -1]
        mas = mas[mas != -1]
        jobs = jobs[jobs != -1]

        # update available combinations
        for i, act_batch_idx in enumerate(active_idxes):
            other_or_opes = torch.where(self.ope_or_batch[act_batch_idx, opes[i]] == 1)[0]
            other_or_opes = other_or_opes[other_or_opes != opes[i]]
            if other_or_opes.numel():
                comb_ids = torch.where(self.combs_id_batch[act_batch_idx, jobs[i]] == 1)[0]
                discard = torch.tensor([1 if tensor.numel() else 0 for tensor in [torch.where(combination[other_or_opes] == 1)[0]
                                                                                for combination in self.combs_batch[act_batch_idx, comb_ids]]])
                self.combs_id_batch[act_batch_idx, jobs[i], comb_ids] -= discard
                job_start_ope = self.num_ope_biases_batch[act_batch_idx, jobs[i]]
                job_end_ope = self.end_ope_biases_batch[act_batch_idx, jobs[i]]
                opes_on_remain_combs = torch.matmul(self.combs_id_batch[act_batch_idx, jobs[i], comb_ids].unsqueeze(0),
                                                    self.combs_batch[act_batch_idx, comb_ids]).squeeze().bool()
                self.remain_opes_batch[act_batch_idx, job_start_ope:job_end_ope + 1] *= opes_on_remain_combs[job_start_ope:job_end_ope + 1]
                if torch.where(self.combs_id_batch[act_batch_idx, jobs[i]] == 1)[0].numel() == 0:
                    raise ValueError("No available combinations")
            


        proc_times = self.proc_times_batch[active_idxes, opes, mas]
        # update feasible opes, for the selected O-M pairs
        self.job_end_time_batch[active_idxes, jobs] = self.time[active_idxes] + proc_times
        opes_matrix = torch.zeros_like(self.ope_req_num_batch, dtype=torch.int64)
        opes_matrix[active_idxes, opes] = 1
        opes_add_matrix = (opes_matrix[active_idxes].unsqueeze(-2).float().\
            matmul(self.ope_pre_adj_batch[active_idxes].float())).int().squeeze(-2)
        self.ope_req_num_batch[active_idxes] -= opes_add_matrix

        # update operation eligible time vector
        self.ready_time[active_idxes] = torch.where(torch.logical_and(self.ope_req_num_batch[active_idxes] == 0, opes_add_matrix == 1),\
            (self.time[active_idxes] + proc_times).unsqueeze(-1).expand_as(self.ready_time[active_idxes]),\
                self.ready_time[active_idxes])
        
        if self.ope_req_num_batch.min() < 0:
            raise ValueError("The number of required operations is less than 0")
          
        self.next_ope_eligible_batch[active_idxes] = self.next_ope_eligible_batch[active_idxes] + \
            torch.where(self.ope_req_num_batch[active_idxes]==0, opes_add_matrix, torch.tensor(0, dtype=torch.int32)) - \
            (opes_matrix[active_idxes].unsqueeze(-2).float().matmul(self.ope_or_batch[active_idxes].float())).int().squeeze(-2)


        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[active_idxes, mas] = 1
        self.ope_ma_adj_batch[active_idxes, opes] = remain_ope_ma_adj[active_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        

        self.info_ope_status_batch[active_idxes, opes] = torch.ones(active_idxes.size(0), dtype=torch.float)
        self.info_ope_proc_time_batch[active_idxes, opes] = proc_times

        self.info_ope_scheduled_start_batch[active_idxes, opes] = self.time[active_idxes]

        self.schedules_batch[active_idxes, opes, :2] = torch.stack((torch.ones(active_idxes.size(0)), mas), dim=1)
        self.schedules_batch[active_idxes, opes, 2] = self.info_ope_scheduled_start_batch[active_idxes, opes]
        self.schedules_batch[active_idxes, opes, 3] = self.info_ope_scheduled_start_batch[active_idxes, opes] + \
                                                       self.info_ope_proc_time_batch[active_idxes, opes]
        self.machines_batch[active_idxes, mas, 0] = torch.zeros(active_idxes.size(0))
        self.machines_batch[active_idxes, mas, 1] = self.time[active_idxes] + proc_times
        self.machines_batch[active_idxes, mas, 2] += proc_times
        self.machines_batch[active_idxes, mas, 3] = jobs.float()

        utiliz = self.machines_batch[batch_idxes, :, 2]
        cur_time = self.time[batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[batch_idxes, None] + 1e-9)

        # Update other variable according to actions
        self.ope_step_batch[active_idxes, jobs] = opes


        self.mask_job_procing_batch[active_idxes, jobs] = torch.where(proc_times > 1e-3, True, False)
        self.mask_ma_procing_batch[active_idxes, mas] = torch.where(proc_times > 1e-3, True, False)

        lag_mask_job_finish_batch = copy.deepcopy(self.mask_job_finish_batch)
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch, True, self.mask_job_finish_batch)
        
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        # get reward
        bonus = torch.zeros(len(active_idxes))
        if self.reward_info['balance_bonus']:
            # active_ave_job_end_batch = torch.sum(self.job_estimate_end_batch[active_idxes] * ~lag_mask_job_finish_batch[active_idxes], dim=1) / torch.sum(~lag_mask_job_finish_batch[active_idxes], dim=1)
            # bonus = (torch.sigmoid(2 * (self.job_estimate_end_batch[active_idxes, jobs] - active_ave_job_end_batch) /
            #                                          (active_ave_job_end_batch * 1.25 - self.time[active_idxes]) * proc_times.bool()) - 0.5) * 2
            bonus = torch.where(self.job_estimate_end_batch[active_idxes, jobs] >= 0.8 * torch.max(self.job_estimate_end_batch[active_idxes] * ~lag_mask_job_finish_batch[active_idxes], dim=1)[0],
                                self.ave_proc_time / 10, - self.ave_proc_time / (10 * self.num_jobs)) * proc_times.bool()
        
        if not self.is_greedy:
            max_makespan = self.get_estimate_end_time(lag_mask_job_finish_batch, wait_idxes)
        else:
            max_makespan = torch.max(self.job_end_time_batch[batch_idxes], dim = 1)[0]
        self.reward_batch = self.makespan_batch - max_makespan
        self.reward_batch[active_idxes] += bonus


        self.makespan_batch = max_makespan

        # print(f"Reward: {self.reward_batch}")
        if self.N % self.change_interval == 0:
                self.proc_time_change(lag_mask_job_finish_batch, actions, ratio_range = 0.25)
        
        

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        wait_idxes_onehot = torch.zeros(self.batch_size, dtype=torch.float)
        wait_idxes_onehot[batch_idxes] = 1
        wait_idxes_onehot[active_idxes] = 0
        
        flag_trans_2_next_time = self.if_no_eligible()
        flag_need_trans = (flag_trans_2_next_time == 0) & ~self.done_batch
        
        self.next_time(wait_idxes_onehot)

        
        while not torch.all(~flag_need_trans):
            self.next_time(flag_need_trans)
            flag_trans_2_next_time = self.if_no_eligible()
            flag_need_trans = (flag_trans_2_next_time == 0) & ~self.done_batch
        
        
        # Update the vector for uncompleted instances
        mask_finish = self.done_batch
        mask_batch_idxes = mask_finish[batch_idxes]
        if mask_finish.any():
            self.batch_idxes = torch.arange(self.batch_size)[~mask_finish]
            
        if not self.done_batch.all():
            if not self.is_greedy:
                self.update_graph(actions[:, ~mask_batch_idxes])
        
        # Update state of the environment
        self.update_state()
        
        info = {}

        return self.state, self.reward_batch, self.done_batch, info

    def get_estimate_end_time(self, lag_mask_job_finish_batch, wait_idxes = torch.tensor([], dtype = torch.int64)):
        batch_idxes = self.batch_idxes
        batch_size = len(batch_idxes)
        time = self.time

        real_wait_idxes = batch_idxes[wait_idxes]
        combs_id_batch = self.combs_id_batch[batch_idxes]
        job_last_ope_batch = self.ope_step_batch[batch_idxes]
        job_last_ope_end_time_batch = self.info_ope_scheduled_start_batch[batch_idxes.unsqueeze(1), job_last_ope_batch] + \
            self.info_ope_proc_time_batch[batch_idxes.unsqueeze(1), job_last_ope_batch]
        job_last_ope_end_time_batch = torch.where(torch.logical_or(job_last_ope_end_time_batch > time[batch_idxes].unsqueeze(1),
                                                                self.mask_job_finish_batch[batch_idxes]),
                                                job_last_ope_end_time_batch,
                                                time[batch_idxes].unsqueeze(1))
        proc_time_batch = torch.zeros(size = (batch_size, self.num_opes)).scatter(1, job_last_ope_batch, job_last_ope_end_time_batch) + \
            (1 - self.info_ope_status_batch[batch_idxes]) * self.info_ope_proc_time_batch[batch_idxes]
        
        if wait_idxes.numel() > 0:
            wait_last_ope_end_time_batch = torch.where(self.mask_job_finish_batch[real_wait_idxes], torch.inf,
                                                    proc_time_batch[wait_idxes.unsqueeze(1),job_last_ope_batch[wait_idxes]])
            time_distance_batch = wait_last_ope_end_time_batch - time[real_wait_idxes].unsqueeze(1)
            # print(time_distance_batch)
            wait_idx, wait_ope = torch.where(time_distance_batch <= 0)
            wait_time, _ = torch.min(torch.where(time_distance_batch > 0, time_distance_batch, torch.inf), dim = 1)
            proc_time_batch[wait_idxes[wait_idx], self.ope_step_batch[real_wait_idxes[wait_idx], wait_ope]] += torch.gather(wait_time, 0, wait_idx)
        combs_time_batch = torch.bmm(proc_time_batch.unsqueeze(1), self.combs_batch[batch_idxes].permute(0, 2, 1))
        job_estimate_end_batch = torch.sum(combs_id_batch * combs_time_batch, dim=2) / torch.sum(combs_id_batch, dim=2) \
            if self.reward_info['comb_mean'] else torch.min(torch.where(combs_id_batch == 1, combs_id_batch * combs_time_batch, torch.inf), dim = 2)[0]
            
        self.job_estimate_end_batch[batch_idxes] = job_estimate_end_batch
        self.combs_time_batch[self.batch_idxes] = combs_time_batch.squeeze(1)
        
        if self.reward_info['estimate']:
            estimate_end_batch = torch.sum(job_estimate_end_batch * ~lag_mask_job_finish_batch[batch_idxes], dim=1) / torch.sum(~lag_mask_job_finish_batch[batch_idxes], dim=1) \
                if self.reward_info['job_mean'] else torch.max(job_estimate_end_batch, dim = 1)[0]
        else:
            estimate_end_batch, _ = torch.max(self.job_end_time_batch[batch_idxes], dim = 1)

        if batch_size == self.batch_size:
            makespan_batch = estimate_end_batch
        else:
            makespan_batch = copy.deepcopy(self.makespan_batch)
            makespan_batch[batch_idxes] = estimate_end_batch

        if torch.isnan(makespan_batch).any():
            raise ValueError("Nan in makespan")
        return makespan_batch

    def find_eligible_pairs(self, find_future = False):
        batch_idxes = self.batch_idxes
        eligible_proc = self.ope_ma_adj_batch[batch_idxes]
        
        ma_eligible = ~self.mask_ma_procing_batch[self.batch_idxes].unsqueeze(1).expand_as(self.ope_ma_adj_batch[self.batch_idxes])
        job_ope_eligible = self.find_job_ope_eligible().unsqueeze(-1).expand_as(self.ope_ma_adj_batch[self.batch_idxes]).bool() 
        opes_eligible = self.find_next_ope_eligible().unsqueeze(-1).expand_as(self.ope_ma_adj_batch[self.batch_idxes])
        
        ope_proc_eligible = opes_eligible & (eligible_proc == 1)
        if not find_future:
            return ma_eligible & ope_proc_eligible & job_ope_eligible
        else:
            return ma_eligible & ope_proc_eligible & job_ope_eligible, ope_proc_eligible & (~ma_eligible | ~job_ope_eligible)
        
    def find_job_ope_eligible(self):
        job_eligible = ~(self.mask_job_procing_batch[self.batch_idxes] + self.mask_job_finish_batch[self.batch_idxes])
        job_ope_eligible = job_eligible.gather(1, self.opes_appertain_batch[self.batch_idxes])
        return job_ope_eligible
    
    def find_next_ope_eligible(self):
        return self.next_ope_eligible_batch[self.batch_idxes].bool()
        
    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        batch_idxes = self.batch_idxes
        eligible_pairs = self.find_eligible_pairs()
        flag_trans_2_next_time = torch.full(size=(self.batch_size,), fill_value=0, dtype=torch.float)
        flag_trans_2_next_time[batch_idxes] = torch.sum(eligible_pairs.transpose(1, 2), dim=[1, 2]).float()
        return flag_trans_2_next_time
    
    def wait_noneligible(self):
        machine_avail_time = self.machines_batch[:, :, 1]
        # available_time of jobs
        job_avail_time = self.job_end_time_batch
        # remain available_time greater than current time
        expanded_time = self.time.unsqueeze(1)  # shape: (batch_size, 1)

        ma_jump = machine_avail_time > expanded_time
        job_jump = job_avail_time > expanded_time

        eligible_index = (torch.any(ma_jump, dim=1) | torch.any(job_jump, dim=1)).view(-1)
        
        return torch.where(eligible_index[self.batch_idxes], 0, 1)

    def next_time(self, flag_need_trans):
        '''
        Transit to the next time
        '''
        # available_time of machines
        machine_avail_time = self.machines_batch[:, :, 1]
        # available_time of jobs
        job_avail_time = self.job_end_time_batch
        # remain available_time greater than current time
        expanded_time = self.time.unsqueeze(1)  # shape: (batch_size, 1)

        ma_jump = machine_avail_time > expanded_time
        job_jump = job_avail_time > expanded_time
        
        flag_need_trans = flag_need_trans.bool() & (torch.any(ma_jump, dim=1) | torch.any(job_jump, dim=1))
        if not torch.any(flag_need_trans):
            return
        trans_idxes = torch.where((flag_need_trans == 1))[0]
        # Calculate the minimum available time of each batch
        machine_greater_than_current = torch.where(ma_jump, machine_avail_time, torch.tensor(float('inf')).to(machine_avail_time.device))
        job_greater_than_current = torch.where(job_jump, job_avail_time, torch.tensor(float('inf')).to(job_avail_time.device))
        
        # Find the minimum value of available_time
        min_machine_time = torch.min(machine_greater_than_current, dim=1, keepdim=True)[0]  # shape: (batch_size, 1)
        min_job_time = torch.min(job_greater_than_current, dim=1, keepdim=True)[0]  # shape: (batch_size, 1)

        # Return the minimum value of available_time (the time to transit to)+
        min_avail_time = torch.min(min_machine_time, min_job_time)  # shape: (batch_size, 1)
        
        # Detect the machines that completed (at above time)
        d = torch.where((machine_avail_time == min_avail_time) & (self.machines_batch[:, :, 0] == 0) & (flag_need_trans[:, None] == 1), True, False)
        # The time for each batch to transit to or stay in
        self.time[trans_idxes] = min_avail_time.squeeze(-1)[trans_idxes]
        self.remain_opes_batch[trans_idxes] *= torch.logical_or(self.schedules_batch[trans_idxes, :, 3] > self.time[trans_idxes].unsqueeze(1),
                                                                 1 - self.info_ope_status_batch[trans_idxes])

        # Update partial schedule (state), variables and infoure vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch,
                                                 True, self.mask_job_finish_batch)
        
    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.num_opes = self.ori_num_opes
        self.num_jobs = self.ori_num_jobs
        self.num_mas = self.ori_num_mas
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)

        # info_opes_batch
        self.info_ope_status_batch = copy.deepcopy(self.old_info_ope_status_batch)
        self.info_ope_proc_time_batch = copy.deepcopy(self.old_info_ope_proc_time_batch)
        self.info_ope_scheduled_start_batch = copy.deepcopy(self.old_info_ope_scheduled_start_batch)
        self.remain_opes_batch = copy.deepcopy(self.old_remain_opes_batch)
        if self.is_add_job:
            self.is_add_job = False
            self.ope_pre_adj_batch = copy.deepcopy(self.old_ope_pre_adj_batch)
            self.opes_appertain_batch = copy.deepcopy(self.old_opes_appertain_batch)
            self.num_ope_biases_batch = copy.deepcopy(self.old_num_ope_biases_batch)
            self.nums_ope_batch = copy.deepcopy(self.old_nums_ope_batch)
            self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
            self.ope_or_batch = copy.deepcopy(self.old_ope_or_batch)
            self.combs_batch = copy.deepcopy(self.old_combs_batch)
        

        self.state = copy.deepcopy(self.old_state)
        self.graph = copy.deepcopy(self.old_graph)
        self.state.graph.data._slice_dict = copy.deepcopy(self.old_edge_indices)
        self.graph.data._slice_dict = copy.deepcopy(self.old_edge_indices)
        # combination info
        self.combs_id_batch = copy.deepcopy(self.old_combs_id_batch)
        self.combs_time_batch = copy.deepcopy(self.old_comb_time_batch)

        self.ready_time = torch.zeros(size = (self.batch_size, self.num_opes))
        self.next_ope_eligible_batch = copy.deepcopy(self.old_next_ope_eligible_batch)
        self.ope_req_num_batch = copy.deepcopy(self.old_ope_req_num_batch)
        self.job_end_time_batch = torch.zeros(size=(self.batch_size, self.num_jobs))

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = 0
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))
        
        # self.makespan_batch = torch.zeros(self.batch_size)
        self.makespan_batch = self.get_estimate_end_time(self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        return self.state
    
    def proc_time_change(self, lag_mask_job_finish_batch, actions, ratio_range = 0.9):
        print("Proc times change.")
        wait_idxes = torch.where(actions[0,:] == -1)[0]

        # Normal
        normal_disturbance  = torch.randn(self.proc_times_batch[self.batch_idxes].size()) * ratio_range / 3 * (1 - self.info_ope_status_batch[self.batch_idxes].unsqueeze(2))
        new_proc_times_batch = (1 + torch.clamp(normal_disturbance, -ratio_range, ratio_range)) * self.proc_times_batch[self.batch_idxes]

        # new proc time info for ope
        new_info_ope_proc_time_batch = torch.sum(new_proc_times_batch, dim=2).div(torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes], dim=2).float() + 1e-9) \
            if self.reward_info['ma_mean'] else torch.min(torch.where(self.ope_ma_adj_batch[self.batch_idxes] == 1, new_proc_times_batch, torch.inf), dim = 2)[0]
        new_info_ope_proc_time_batch = torch.where(new_info_ope_proc_time_batch == torch.inf, 0, new_info_ope_proc_time_batch)
        self.info_ope_proc_time_batch[self.batch_idxes] = torch.where(self.info_ope_status_batch[self.batch_idxes].bool(), self.info_ope_proc_time_batch[self.batch_idxes], new_info_ope_proc_time_batch)

        self.makespan_batch = self.get_estimate_end_time(lag_mask_job_finish_batch, wait_idxes = wait_idxes)


    def add_job(self, case = None, add_batch_idxes = None, data_source = 'case'):
        '''
        Add a job to the environment
        PARAMS:
            case: The job to be added, default is None
            data_source: The source type of the job
        '''

        if ~self.is_add_job:
            self.is_add_job = True
            self.old_ope_pre_adj_batch = copy.deepcopy(self.ope_pre_adj_batch)
            self.old_opes_appertain_batch = copy.deepcopy(self.opes_appertain_batch)
            self.old_num_ope_biases_batch = copy.deepcopy(self.num_ope_biases_batch)
            self.old_nums_ope_batch = copy.deepcopy(self.nums_ope_batch)
            self.old_ope_or_batch = copy.deepcopy(self.ope_or_batch)
            self.old_combs_batch = copy.deepcopy(self.combs_batch)

        lines = []
        if isinstance(add_batch_idxes, torch.Tensor):
            add_batch_idxes = add_batch_idxes.tolist()
        if data_source=='case':  # Generate instances through generators
            for i in range(len(add_batch_idxes)):
                lines.append(case.get_case())  # Generate an instance and save it
                # Records the maximum number of operations in the parallel instances
        
        elif data_source=='tensor':  # Load instances from tensors
            tensor_loaded = case['tensor']
            infos = case['info']

        
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.read().splitlines()
                    lines.append(line)

                
        # load feats and paths
        num_data = 13
        tensors = [[] for _ in range(num_data)]

        new_valid_opes_num = torch.zeros(size=(len(add_batch_idxes),), dtype=torch.long)
        new_valid_mas_num = torch.zeros(size=(len(add_batch_idxes),), dtype=torch.long)
        new_valid_jobs_num = torch.zeros(size=(len(add_batch_idxes),), dtype=torch.long)
        new_valid_combs_num = torch.zeros(size=(len(add_batch_idxes),), dtype=torch.long)
        current_graph_list = self.graph.data.to_data_list()
        for i in tqdm(range(len(add_batch_idxes)),desc="Loading graph features and paths"):
            
            if data_source=='tensor':
                load_data = tensor_loaded[i]
            else:
                load_data = load_ipps(lines[i])

            for j in range(num_data):
                tensors[j].append(load_data[j])
            new_graph = self.graph.cat_load_features(load_data, current_graph_list[add_batch_idxes[i]])
            new_valid_combs_num[i] = new_graph['combs'].x.size(0)
            new_valid_jobs_num[i] = new_graph['jobs'].x.size(0)
            new_valid_opes_num[i] = new_graph['opes'].x.size(0)
            new_valid_mas_num[i] = new_graph['mas'].x.size(0)
            current_graph_list[add_batch_idxes[i]] = new_graph
            
        self.num_opes = new_valid_opes_num.max().item()
        self.num_mas = new_valid_mas_num.max().item()
        self.num_jobs = new_valid_jobs_num.max().item()
        self.num_combs = new_valid_combs_num.max().item()
        add_mas_num = self.num_mas - self.valid_mas_num.max().item()
        graph_list = current_graph_list
        old_valid_mas_num = torch.zeros_like(self.valid_mas_num)        
        add_batch_idxes = torch.tensor(add_batch_idxes, dtype=torch.long)
        self.combs_id_batch = pad_stack_add_idxes(self.combs_id_batch, tensors[10], (self.valid_jobs_num, self.valid_combs_num), (new_valid_jobs_num, new_valid_combs_num), add_batch_idxes, value=0, dim=2)
        # # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = pad_stack_add_idxes(self.proc_times_batch, tensors[0], (self.valid_opes_num, old_valid_mas_num), (new_valid_opes_num, new_valid_mas_num), add_batch_idxes, value=0, dim=2).float()
        # # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = pad_stack_add_idxes(self.ope_ma_adj_batch, tensors[1], (self.valid_opes_num, old_valid_mas_num), (new_valid_opes_num, new_valid_mas_num), add_batch_idxes, value=0, dim=2).long()
        # # shape: (batch_size, num_opes)
        self.remain_opes_batch = pad_stack_add_idxes(self.remain_opes_batch, tensors[12], self.valid_opes_num, new_valid_opes_num, add_batch_idxes, value=0, dim=1).long()
        # # static feats
        # # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = pad_stack_add_idxes(self.ope_pre_adj_batch, tensors[2], (self.valid_opes_num, self.valid_opes_num), (new_valid_opes_num, new_valid_opes_num), add_batch_idxes, value=0, dim=2).long()
        # # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = pad_stack_add_idxes(self.opes_appertain_batch, [ts + torch.max(self.opes_appertain_batch, dim = 1)[0][add_batch_idxes[i]] + 1 for i, ts in enumerate(tensors[3])],
                                                        self.valid_opes_num, new_valid_opes_num, add_batch_idxes, value=0, dim=1).long()
        # # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = pad_stack_add_idxes(self.num_ope_biases_batch, [ts + torch.sum(self.nums_ope_batch[add_batch_idxes[i]]) for i, ts in enumerate(tensors[4])],
                                                        self.valid_jobs_num, new_valid_jobs_num, add_batch_idxes, value=0, dim=1).long()
        # # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = pad_stack_add_idxes(self.nums_ope_batch, tensors[5], self.valid_jobs_num, new_valid_jobs_num, add_batch_idxes, value=0, dim=1).long()
        # # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # # shape: (batch_size, num_opes, num_opes), whether exist OR between operations
        self.ope_or_batch = pad_stack_add_idxes(self.ope_or_batch, tensors[7], (self.valid_opes_num, self.valid_opes_num), (new_valid_opes_num, new_valid_opes_num), add_batch_idxes, value=0, dim=2).long()
        # # shape: (batch_size, num_combs, num_opes)
        self.combs_batch = pad_stack_add_idxes(self.combs_batch, tensors[11], (self.valid_combs_num, self.valid_opes_num), (new_valid_combs_num, new_valid_opes_num), add_batch_idxes, value=0, dim=2).float()

        self.ope_step_batch = pad_stack_add_idxes(self.ope_step_batch, [ts + torch.sum(self.nums_ope_batch[add_batch_idxes[i], :-1]) for i, ts in enumerate(tensors[4])],
                                                  self.valid_jobs_num, new_valid_jobs_num, add_batch_idxes, value=-1, dim=1).long()
        # # shape: (batch_size, num_opes) Whether the operation is eligible
        self.next_ope_eligible_batch = pad_stack_add_idxes(self.next_ope_eligible_batch, tensors[9], self.valid_opes_num, new_valid_opes_num, add_batch_idxes, value=0, dim=1).long()
        # # shape: (batch_size, num_opes), the number of required operations for each operation
        self.ope_req_num_batch = pad_stack_add_idxes(self.ope_req_num_batch, tensors[8], self.valid_opes_num, new_valid_opes_num, add_batch_idxes, value=100, dim=1).long()
        # # shape: (batch_size, num_jobs), the completion time of each job
        self.job_end_time_batch = pad_to_given_size(self.job_end_time_batch, self.nums_ope_batch.size(1), value = 0, num_dims=1)
        # # shape: (batch_size, num_opes), the ready time of each operation
        self.ready_time = pad_to_given_size(self.ready_time, self.remain_opes_batch.size(1), value = 0, num_dims=1)
        # # shape: (batch_size, num_combs) 
        self.combs_time_batch = pad_to_given_size(self.combs_time_batch, self.combs_batch.size(1), value = 0, num_dims=1)
       
        self.info_ope_status_batch = pad_to_given_size(self.info_ope_status_batch, self.num_opes, value = 0, num_dims=1).scatter(1, self.num_ope_biases_batch, 1)
        
        new_info_ope_proc_time_batch = torch.sum(self.proc_times_batch, dim=2).div(torch.count_nonzero(self.ope_ma_adj_batch, dim=2).float() + 1e-9) \
            if self.reward_info['ma_mean'] else torch.min(torch.where(self.ope_ma_adj_batch == 1, self.proc_times_batch, torch.inf), dim = 2)[0]
        new_info_ope_proc_time_batch = torch.where(new_info_ope_proc_time_batch == torch.inf, 0, new_info_ope_proc_time_batch)
        
        self.info_ope_proc_time_batch = pad_to_given_size(self.info_ope_proc_time_batch, self.num_opes, value = 0, num_dims=1)
        self.info_ope_proc_time_batch[self.batch_idxes] = torch.where(self.info_ope_status_batch[self.batch_idxes].bool(), self.info_ope_proc_time_batch[self.batch_idxes], new_info_ope_proc_time_batch[self.batch_idxes])
        
        self.info_ope_scheduled_start_batch = pad_to_given_size(self.info_ope_scheduled_start_batch, self.num_opes, value = 0, num_dims=1)

        self.mask_job_procing_batch = pad_to_given_size(self.mask_job_procing_batch, self.num_jobs, value = False, num_dims=1)

        self.mask_job_finish_batch = pad_to_given_size(self.mask_job_finish_batch, self.num_jobs, value = False, num_dims=1)
        self.mask_job_finish_batch = torch.where(self.end_ope_biases_batch == -1, True, self.mask_job_finish_batch)

        self.mask_ma_procing_batch = pad_to_given_size(self.mask_ma_procing_batch, self.num_mas, value = False, num_dims=1)

        self.schedules_batch = pad_to_middle_given_size(self.schedules_batch, self.num_opes, value = 0)

        self.machines_batch = pad_to_middle_given_size(self.machines_batch, self.num_mas, value = 0)
        self.machines_batch[:, self.num_mas - add_mas_num:, 0] = 1 
        
        self.makespan_batch = self.get_estimate_end_time(self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.graph.init_from_graph_list(graph_list, self.combs_time_batch, self.remain_opes_batch, self.job_estimate_end_batch)
        
        self.init_state()

        # Save initial data for reset
        self.valid_opes_num[add_batch_idxes] = new_valid_opes_num
        self.valid_mas_num[add_batch_idxes] = new_valid_mas_num
        self.valid_jobs_num[add_batch_idxes] = new_valid_jobs_num
        self.valid_combs_num[add_batch_idxes] = new_valid_combs_num
        
            

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        RETURN:
            True: The schedule is feasible
            False: The schedule is infeasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for _ in range(self.batch_size)]
        nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(nums_opes[batch_id])):
                step = schedules[i]
                if step[0].item() == 1:
                    ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: (s[2], s[1]))
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if (ma_gantt[i][j][2] - ma_gantt[i][j][1] - proc_time[ma_gantt[i][j][0]][i]) > 1e-4:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        flag_wrong_scheduled = 0
        ope_sub_adj_batch = self.ope_pre_adj_batch.permute(0, 2, 1)
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            ope_req_num = self.ope_req_num_batch[k]
            for i in range(self.num_jobs):
                if self.end_ope_biases_batch[k, i] == -1:
                    continue
                acted_ope = [[j, schedule[j][2].item(), schedule[j][3].item()] 
                                   for j in range(num_ope_biases[i], num_ope_biases[i] + nums_ope[i]) if schedule[j][0] != 0]
                acted_ope = sorted(acted_ope, key=lambda x: (x[2], x[1]))
                acted_ope_index = [act[0] for act in acted_ope]
                flag_wrong_scheduled += sum(ope_req_num[acted_ope_index]).item()
                
                for m in range(1, len(acted_ope)):
                    if acted_ope[m - 1][2] > acted_ope[m][1]:
                        flag_ope_overlap += 1
                
                if acted_ope[len(acted_ope) - 1][0] + 1 == sum(nums_ope):
                    continue
                if torch.any(ope_sub_adj_batch[k][acted_ope[len(acted_ope) - 1][0] + 1]):
                    flag_wrong_scheduled += 1


        # check combiantions
        flag_wrong_comb_num = torch.stack(torch.where(self.combs_id_batch.sum(dim = 2) > 1)).numel()
        flag_wrong_comb_ope = torch.stack(torch.where(torch.bmm(self.combs_id_batch.any(dim=1).float().unsqueeze(1),
                                                                self.combs_batch).squeeze(1) != self.info_ope_status_batch)).numel()
        


        if flag_ma_overlap + flag_proc_time + flag_ope_overlap + flag_wrong_scheduled + flag_wrong_comb_num + flag_wrong_comb_ope != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch
        
    def get_schedule(self, idx):
        # ope, ma, job, start, end
        schedule = np.array([[i, ma, self.opes_appertain_batch[idx, i].item(), start, end] for i, (done, ma, start, end)
                    in enumerate(np.array(self.schedules_batch[idx].to('cpu'))) if done or i in self.num_ope_biases_batch[idx]])
        matrix_cal_cumul = getAncestors(self.ope_pre_adj_batch[idx])
        schedule = sort_schedule(schedule, matrix_cal_cumul)
        return schedule
        

    def close(self):
        pass
