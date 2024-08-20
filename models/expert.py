import sys
sys.path.append(".")
import torch
from models.memory import MemoryIL
from network.utils import *
from utils.utils import find_action_indexes
import torch
import torch.nn.functional as F
import copy
from env.ipps_env import EnvState
from omegaconf import OmegaConf
from models.policy import ExpertPolicy
class Expert(torch.nn.Module):
    """Used to define the expert model in Bahevioral Cloning

    """
    def __init__(
                self, 
                config,
                sols:torch.Tensor = None, 
                policy:ExpertPolicy = None,
                
                ):
        super(Expert, self).__init__()
        if sols is not None:
            self.sols = sols
            self.pointer = torch.zeros(len(sols), dtype=torch.long)
        self.device = config.device
    
        self.policy = policy if policy is not None else ExpertPolicy(config)

    
    def act(
            self, 
            state: EnvState, 
            memory: MemoryIL, 
            flag_sample: bool = True, 
            ):
        return self.policy.act(state, memory, flag_sample)
    
    def give_solution(self, sols):
        self.sols = sols
        self.pointer = torch.zeros(len(sols), dtype=torch.long)
    
    def act_expert(self, state: EnvState, memory: MemoryIL):
        """
        Executes the expert's action selection process based on the given environment state and memory.

        Args:
            state (EnvState): The current environment state.
            memory (MemoryIL): The memory object containing relevant information.

        Returns:
            torch.Tensor: The selected action.

        Raises:
            ValueError: If no eligible pairs are found.

        """
        if self.sols is None:
            raise ValueError("No solution provided")
        with torch.no_grad():
            # Get the eligible pairs
            batch_idxes = state.batch_idxes
            batchsize = state.opes_appertain_batch.size(0)

            eligible = state.eligible_pairs
            if (~(eligible)).all():
                raise ValueError("No eligible pairs")

            _, embs_local_global, ptr_proc, embs_wait, ptr_wait, indicator_wait \
                    = self.policy.get_embedding(state.graph, eligible, state.future_eligible_pairs, state.opes_appertain_batch, batch_idxes)
                        
            if self.policy.wait_deal == 'max':
                _, wait_max_idxes = self.policy.embs_to_probs(embs_local_global, ptr_proc, embs_wait, ptr_wait, \
                    indicator_wait=indicator_wait, return_max_wait_idxes=True)
            elif self.policy.wait_deal == 'mean':
                _ = self.policy.embs_to_probs(embs_local_global, ptr_proc, embs_wait, ptr_wait, \
                    indicator_wait=indicator_wait)

            action = torch.zeros(3, batch_idxes.size(0), dtype=torch.long) - 1
            action = self.sols[batch_idxes, self.pointer[batch_idxes], 0:3].t()


            action_indexes = torch.full((batch_idxes.size(0),), -1, dtype=torch.long, device=self.device)

            wait_idx = torch.nonzero(indicator_wait).squeeze(-1)
            padded_wait_idx = torch.scatter(torch.full((action_indexes.size(0),), -1, dtype=torch.long),
                                            0, wait_idx, torch.arange(wait_idx.size(0), device=self.device))

            padded_backups = [[] for _ in range(batchsize)]
            # Choose the action and get the action embedding
            for i in range(action.size(1)):
                eligible_pair = eligible[i].nonzero().t()
                find_act = find_action_indexes(eligible_pair, action[0:2, i])
                if find_act is None:
                    action[:, i] = torch.Tensor([-1, -1, -1])
                    action_indexes[i] = -1
                    index = padded_wait_idx[i]
                    assert index != -1
                    
                else:

                    action_indexes[i] = find_act
                    self.pointer[state.batch_idxes[i]] += 1


            # Update the memory
            padded_action_indexes = torch.zeros(batchsize, dtype=torch.long, device=self.device)
            padded_action_indexes[batch_idxes] = action_indexes
            padded_indices = torch.zeros(batchsize, device=self.device, dtype=torch.bool)
            padded_indices[state.batch_idxes] = 1
            padded_eligible = torch.zeros((batchsize, *eligible.size()[1:]), device=self.device, dtype=eligible.dtype)
            padded_eligible[state.batch_idxes] = eligible
            padded_future_eligible = torch.zeros((batchsize, *state.future_eligible_pairs.size()[1:]), device=self.device, dtype=state.future_eligible_pairs.dtype)
            padded_future_eligible[state.batch_idxes] = state.future_eligible_pairs

            memory.eligible.append(padded_eligible)
            memory.graphs.append(copy.deepcopy(state.graph.data.to_data_list()))
            memory.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memory.actives.append(padded_indices)
            memory.action_indexes.append(padded_action_indexes)
            memory.future_eligible.append(padded_future_eligible)
            memory.opes_appertain.append(state.opes_appertain_batch)
            memory.backup_act.append(padded_backups)

        return action
    
    def evaluate(self, graph, eligible, future_eligible, opes_appertain, actions, batch_idxes, backup_act=None):
        """
        Evaluates the loss for the given inputs.

        Args:
            graph (Tensor): The input graph.
            eligible (Tensor): The eligible nodes.
            future_eligible (Tensor): The future eligible nodes.
            opes_appertain (Tensor): The OPES appertain nodes.
            actions (Tensor): The actions taken.
            batch_idxes (Tensor): The batch indexes.
            backup_act (Tensor, optional): The backup actions. Defaults to None.

        Returns:
            Tensor: The loss value.
        """
        if self.sols is None:
            raise ValueError("No solution provided")

        _, embs_local_global, ptr, embs_wait, ptr_wait, indicator_wait = self.policy.get_embedding(
            graph, eligible, future_eligible, opes_appertain, batch_idxes
        )

        action_log_probs = self.policy.embs_to_probs(
            embs_local_global, ptr, embs_wait, ptr_wait, indicator_wait, log_probs=True
        )
        inf_mask = (action_log_probs == float("-inf"))
        action_log_probs = action_log_probs.masked_fill(inf_mask, -1e10)

        wait_indice = action_log_probs.size(-1) - 1
        actions = torch.where(actions == -1, wait_indice, actions)
        actions_one_hot = F.one_hot(actions.long(), action_log_probs.size(-1)).float()

        loss = F.kl_div(action_log_probs, actions_one_hot, reduction="batchmean")

        return loss
        
        
        
if __name__ == "__main__":
    sols = ["problemsss/ortools_schedule/o2d_sol_problem_job_0_15_611_729.ipps", "problemsss/ortools_schedule/o2d_sol_problem_job_1_11_333_352_419_509_698.ipps"]
    train_paras = "cpu"
    config = OmegaConf.load("config.yaml")
    expert = Expert(sols, train_paras,config.nn_paras)
    print(expert.sols)
    print(expert.pointer)
    print(expert.device)
    print("Done!")