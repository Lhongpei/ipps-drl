import torch
import torch.nn as nn
import copy
import math
import itertools
from network.hetero_data import Graph_Batch
from models.policy import DRLPolicy
class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm implementation.

    Args:
        model_paras (dict): Model parameters.
        train_paras (dict): Training parameters.
        num_envs (int, optional): Number of parallel instances. Defaults to None.

    Attributes:
        lr (float): Learning rate.
        betas (tuple): Default value for Adam optimizer.
        gamma (float): Discount factor.
        eps_clip (float): Clip ratio for PPO.
        K_epochs (int): Number of epochs to update the policy.
        A_coeff (float): Coefficient for policy loss.
        vf_coeff (float): Coefficient for value loss.
        entropy_coeff (float): Coefficient for entropy term.
        num_envs (int): Number of parallel instances.
        device (str): PyTorch device.
        policy (DRLPolicy): Actor-Critic agent.
        policy_old (DRLPolicy): Copy of the old policy.
        optimizer (torch.optim.Adam): Adam optimizer.
        MseLoss (torch.nn.MSELoss): Mean Squared Error loss.

    Methods:
        update(memory, env_paras, train_paras): Update the policy using the collected memory.

    """

    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras.lr  # learning rate
        self.betas = train_paras.betas  # default value for Adam
        self.gamma = train_paras.gamma  # discount factor
        self.eps_clip = train_paras.eps_clip  # clip ratio for PPO
        self.clip_multi = train_paras.clip_multi
        self.clip_ub = train_paras.clip_ub
        self.K_epochs = train_paras.K_epochs  # Update policy for K epochs
        self.A_coeff = train_paras.A_coeff  # coefficient for policy loss
        self.vf_coeff = train_paras.vf_coeff  # coefficient for value loss
        self.entropy_coeff = train_paras.entropy_coeff  # coefficient for entropy term
        self.entropy_dicount = train_paras.entropy_discount
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras.device  # PyTorch device

        self.policy = DRLPolicy(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()
    
    def load_pretrained_policy(self, pretrained_policy_path):
        self.policy.load_actor(pretrained_policy_path)
        self.policy_old.load_actor(pretrained_policy_path)
        print(f'Pretrained policy loaded successfully from {pretrained_policy_path}')
    
    def update(self, memory, env_paras, train_paras):
        """
        Update the policy using the collected memory.

        Args:
            memory (MemoryRL): MemoryRL object containing the collected data.
            env_paras (dict): Environment parameters.
            train_paras (dict): Training parameters.

        Returns:
            float: A dictionary containing the loss values.
            float: Average discounted rewards per timestep.

        """
        device = env_paras.device
        minibatch_size = train_paras.minibatch_size  # batch size for updating

        old_actives = torch.stack(memory.actives, dim=0).transpose(0, 1).flatten(0, 1)
        old_actives_no_flat = torch.stack(memory.actives, dim=0).transpose(0, 1)

        # Deal with Graph
        memory_graphs = list(itertools.chain(*zip(*memory.graphs)))
        selected_graphs = [memory_graphs[i] for i in old_actives.nonzero(as_tuple=True)[0]]

        # Ensure all tensors are float
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1).float()
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0, 1).float()
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        old_waits = torch.stack(memory.waits, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        old_future_eligible = torch.stack(memory.future_eligible, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        old_opes_appertain = torch.stack(memory.opes_appertain, dim=0).transpose(0, 1).flatten(0, 1).float()[old_actives]
        
        if torch.isnan(memory_rewards).any() or torch.isinf(memory_rewards).any():
            print('Memory rewards contain NaN or Inf')
        
        # Estimate and normalize the rewards
        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            active_rewards = memory_rewards[i][old_actives_no_flat[i]].squeeze()  # 仅选取有效的奖励
            active_terminals = memory_is_terminals[i][old_actives_no_flat[i]].squeeze()  # 仅选取有效的终止标志
            for reward, is_terminal in zip(reversed(active_rewards), reversed(active_terminals)):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)  # Convert rewards to float
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) 
            #rewards = rewards/100
            rewards_envs.append(rewards)
        
        rewards_envs = torch.cat(rewards_envs)

        loss_epochs = 0
        policy_loss_epochs = 0
        value_loss_epochs = 0
        entropy_loss_epochs = 0
        full_batch_size = old_actives.nonzero().size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                    
                if start_idx >= end_idx - 1:
                    continue
                
                batch_idxes = torch.arange(start_idx, end_idx).long().to(device)
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    graph = Graph_Batch(selected_graphs[start_idx:end_idx]),
                    eligible = old_eligible[batch_idxes], 
                    future_eligible=old_future_eligible[batch_idxes],
                    opes_appertain=old_opes_appertain[batch_idxes],
                    actions = old_action_envs[batch_idxes], 
                    batch_idxes = torch.arange(batch_idxes.size(0)).long(),
                    waits = old_waits[batch_idxes]
                )

                ratios = torch.exp(logprobs - old_logprobs[start_idx:end_idx].detach())
                advantages = rewards_envs[start_idx:end_idx] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                self.eps_clip = self.eps_clip * self.clip_multi if self.eps_clip * self.clip_multi < self.clip_ub else self.clip_ub
                policy_loss = - torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(state_values.view(-1), rewards_envs[start_idx:end_idx].view(-1))
                entropy_loss = - dist_entropy.mean()
                loss = self.A_coeff * policy_loss \
                       + self.vf_coeff * value_loss \
                       + self.entropy_coeff * entropy_loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print('Loss contains NaN or Inf')
                loss_epochs += loss.detach()
                policy_loss_epochs += policy_loss.detach()
                value_loss_epochs += value_loss.detach()
                entropy_loss_epochs += entropy_loss.detach()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.entropy_coeff  *= self.entropy_dicount
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        loss_dict = {
            "total_loss": loss_epochs.item() / self.K_epochs,
            "policy_loss": policy_loss_epochs.item() / self.K_epochs,
            "value_loss": value_loss_epochs.item() / self.K_epochs,
            "entropy_loss": entropy_loss_epochs.item() / self.K_epochs
        }
        return loss_dict, discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])

