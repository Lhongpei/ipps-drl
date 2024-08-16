import torch
import torch.nn as nn
import copy
import math
import itertools
from network.hetero_data import Graph_Batch
from models.expert import Expert, ExpertPolicy
from utils.utils import solutions_padding
from models.memory import MemoryIL
from tqdm import tqdm
class BehaviorCloning:
    """
    Class representing the behavior cloning algorithm.

    Args:
        config: Configuration object containing the training parameters and neural network parameters.

    Attributes:
        lr (float): Learning rate for the optimizer.
        betas (Tuple[float, float]): Beta values for the Adam optimizer.
        K_epochs (int): Number of epochs to update the policy for.
        device (str): PyTorch device to use for computation.
        minibatch_size (int): Batch size for updating the policy.
        policy (Expert): Expert policy network.
        policy_old (Expert): Copy of the expert policy network.
        optimizer (torch.optim.Adam): Adam optimizer for updating the policy.
        MseLoss (nn.MSELoss): Mean squared error loss function.


    Methods:
        update(memory: MemoryIL) -> float:
            Update the policy using the collected memory.

    """

    def __init__(self, config, sols: torch.Tensor = None):
        self.lr = config.train_paras.lr  # learning rate
        self.betas = config.train_paras.betas  # default value for Adam
        self.K_epochs = config.train_paras.K_epochs  # Update policy for K epochs
        self.device = config.nn_paras.device  # PyTorch device
        self.minibatch_size = config.train_paras.minibatch_size

        self.expert = Expert(config.nn_paras, sols)
        self.optimizer = torch.optim.Adam(self.expert.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def give_solution(self, sols):
        self.expert.give_solution(sols)
    
    def update(
                self, 
                memory: MemoryIL
               ) -> float:
        """
        Update the policy using the collected memory.

        Args:
            memory (MemoryIL): MemoryIL object containing the collected data.

        Returns:
            float: Average loss per epoch.

        """
        device = self.device
        minibatch_size = self.minibatch_size  # batch size for updating

        old_actives = torch.stack(memory.actives, dim=0).transpose(0, 1).flatten(0, 1)

        # Deal with Graph
        memory_graphs = list(itertools.chain(*zip(*memory.graphs)))
        # memory_backup = list(itertools.chain(*zip(*memory.backup_act)))
        selected_graphs = [memory_graphs[i] for i in old_actives.nonzero(as_tuple=True)[0]]

        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        old_future_eligible = torch.stack(memory.future_eligible, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        old_opes_appertain = torch.stack(memory.opes_appertain, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        
        loss_epochs = 0
        full_batch_size = old_actives.nonzero().size(0)
        shuffled_idxes = torch.randperm(full_batch_size)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in tqdm(range(self.K_epochs), desc='Training Epochs '):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                    
                if start_idx >= end_idx - 1:
                    continue
                
                batch_idxes = shuffled_idxes[torch.arange(start_idx, end_idx).long()]
                # batch_idxes = batch_idxes[torch.randperm(batch_idxes.size(0)).to(device)]
                loss = self.expert.evaluate(
                    Graph_Batch([selected_graphs[i] for i in batch_idxes]),
                    old_eligible[batch_idxes], 
                    old_future_eligible[batch_idxes],
                    old_opes_appertain[batch_idxes],
                    old_action_envs[batch_idxes], 
                    torch.arange(batch_idxes.size(0)).long(),
                    # [selected_backups[i] for i in batch_idxes]
                )

                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        return loss_epochs.item() / self.K_epochs

