from network.models import GraphEmbedding, Actor, Critic
from network.utils import *
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
from models.memory import MemoryRL, MemoryIL
from env.ipps_env import EnvState
class Policy(torch.nn.Module):
    """
    A class representing an base policy.

    Attributes:
        graph_embedding (GraphEmbedding): The graph embedding module.
        actor (Actor): The actor module.

    Methods:
    """

    def __init__(self, config):
        """
        Initializes the Policy object.

        Args:
            config: The configuration object.
        """
        super(Policy, self).__init__()
        
        self.wait_deal = config.wait_deal
        self.pooling_method = config.pooling_method

        self.graph_embedding = GraphEmbedding(config)    
        self.actor = Actor(config)
        self.actor4wait = Actor(config)
        self.critic = Critic(config)
        self.qkv = config.qkv
        if self.qkv:
            self.queue_linear = nn.Linear(self.actor.actor[0].in_features, self.actor.actor[0].in_features)
            self.queue4wait_linear = nn.Linear(self.actor.actor[0].in_features, self.actor.actor[0].in_features)
            self.key_linear = nn.Linear(self.actor.actor[0].in_features, self.actor.actor[0].in_features)
            self.key4wait_linear = nn.Linear(self.actor.actor[0].in_features, self.actor.actor[0].in_features)
            self.value_linear = nn.Linear(self.actor.actor[0].in_features, self.actor.actor[0].in_features)
            self.value4wait_linear = nn.Linear(self.actor.actor[0].in_features, self.actor.actor[0].in_features)
        self.device = config.device
        
        
        
    def save_actor(self, path):
        """
        Saves the actor model to the specified path.

        Args:
            path (str): The path to save the actor model.

        Returns:
            None
        """
        state_dict = {}
        
        # Save graph_embedding state_dict
        for k, v in self.graph_embedding.state_dict().items():
            state_dict[f"graph_embedding.{k}"] = v
            
        # Save actor state_dict
        for k, v in self.actor.state_dict().items():
            state_dict[f"actor.{k}"] = v
            
        torch.save(state_dict, path)

    def load_actor(self, path):
        """
        Loads the actor model from the specified path.

        Args:
            path (str): The path to the saved actor model.

        Returns:
            None
        """
        state_dict = torch.load(path)
        self.graph_embedding.load_state_dict({k.replace("graph_embedding.", ""): v for k, v in state_dict.items() \
            if k.startswith("graph_embedding.")})
        self.actor.load_state_dict({k.replace("actor.actor", "actor"): v for k, v in state_dict.items() \
            if k.startswith("actor.")})
        

    def get_embedding_by_eligible(self, graph, eligible, opes_appertain, batch_idxes):
        """
        Get embeddings based on eligibility criteria.

        Args:
            graph (Graph): The input graph.
            eligible (list): List of eligible nodes.
            opes_appertain (list): List of opes appertaining to the eligible nodes.
            batch_idxes (list): List of batch indexes.

        Returns:
            tuple: A tuple containing the following elements:
                - embs_nodes (Tensor): Embeddings of the eligible nodes.
                - embs_global (Tensor): Global embeddings.
                - embs_local_global (Tensor): Concatenation of local and global embeddings.
                - ptr (Tensor): Pointer tensor.

        """
        new_graph = copy.deepcopy(graph.data)
        graph_emb = self.graph_embedding(normalize_hetero_data(new_graph))
        embs_nodes, ptr = get_graph_embs_to_pair(graph_emb, eligible, opes_appertain, batch_idxes)

        embs_global = global_pooling(graph_emb, pool_type=self.pooling_method)[batch_idxes]

        embs_local_global = cat_local_global_embedding(embs_nodes, embs_global, ptr)

        return embs_nodes, embs_global, embs_local_global, ptr,

    
    def get_embedding(self, graph, eligible, future_eligible, opes_appertain, batch_idxes):
        """
        Computes the embeddings for the given inputs.

        Args:
            graph (Tensor): The input graph.
            eligible (Tensor): The eligible tensor.
            future_eligible (Tensor): The future eligible tensor.
            opes_appertain (Tensor): The opes appertain tensor.
            batch_idxes (Tensor): The batch indexes.

        Returns:
            Tuple: A tuple containing the following elements:
                - embs_global_sq (Tensor): The embeddings for the global square.
                - embs_local_global (Tensor): The embeddings for the local global.
                - ptr_proc (Tensor): The pointer for the process.
                - embs_wait (Tensor): The embeddings for the wait.
                - ptr_wait (Tensor): The pointer for the wait.
                - indicator_wait (Tensor): The indicator for the wait.
        """
        _, embs_global_sq, embs_local_global, ptr_proc = self.get_embedding_by_eligible(graph, eligible, opes_appertain, batch_idxes)
        indicator_wait = torch.any(future_eligible, dim=[1,2])
        if not torch.any(indicator_wait):
            embs_wait = None
            ptr_wait = None
        else:
            indicator_wait = torch.any(future_eligible, dim=[1,2])
            wait_eligible_idxes = batch_idxes[indicator_wait]
            _, _, embs_wait, ptr_wait = self.get_embedding_by_eligible(graph, future_eligible[indicator_wait], opes_appertain, wait_eligible_idxes)
        
        return embs_global_sq, embs_local_global, ptr_proc, embs_wait, ptr_wait, indicator_wait
    
    def embs_to_probs(self, embs_local_global, ptr_proc, embs_wait, ptr_wait, indicator_wait, log_probs=False, return_max_wait_idxes=False):
        """
        Converts embeddings to action probabilities.

        Args:
            embs_local_global (torch.Tensor): Embeddings for local and global information.
            ptr_proc (torch.Tensor): Pointer to process embeddings.
            embs_wait (torch.Tensor): Embeddings for waiting information.
            ptr_wait (torch.Tensor): Pointer to wait embeddings.
            indicator_wait (torch.Tensor): Indicator for wait embeddings.
            log_probs (bool, optional): Flag indicating whether to return log probabilities. Defaults to False.
            return_max_wait_idxes (bool, optional): Flag indicating whether to return the indices of maximum wait probabilities. Defaults to False.

        Returns:
            torch.Tensor: Action probabilities.

        Raises:
            ValueError: If there are NaN values in the action probabilities.
        """
        can_wait = getattr(self, 'wait_flag', True)
        active_proc = ptr_proc[1:] - ptr_proc[0:-1]
        len_proc = torch.max(active_proc)
        return_index = None
        if self.qkv:
            eligible_queues = shift_to_batch(self.queue_linear(embs_local_global), ptr_proc)
            eligible_queues_mask = ~eligible_queues.isinf()
            eligible_queues = torch.where(eligible_queues_mask, eligible_queues, torch.zeros_like(eligible_queues))
            eligible_keys = shift_to_batch(self.key_linear(embs_local_global), ptr_proc)
            eligible_keys_mask = ~eligible_keys.isinf()
            eligible_keys = torch.where(eligible_keys_mask, eligible_keys, torch.zeros_like(eligible_keys))
            eligible_values = shift_to_batch(self.value_linear(embs_local_global), ptr_proc, padding_value=0)
            eligible_score_mask = torch.bmm(eligible_queues_mask.float(), eligible_keys_mask.float().permute(0, 2, 1)).bool()
            eligible_score = torch.where(eligible_score_mask, torch.bmm(eligible_queues, eligible_keys.permute(0, 2, 1)), -torch.inf)

            batch_size, matrix_size = eligible_score.shape[0:2]
            eligible_score[:, torch.arange(matrix_size), torch.arange(matrix_size)] = -torch.inf
            eligible_all_inf_mask = (eligible_score == -torch.inf).all(dim=-1, keepdim=True)
            eligible_score = torch.softmax(eligible_score, dim=-1)
            eligible_score = torch.where(eligible_all_inf_mask, torch.zeros_like(eligible_score), eligible_score)
            eligible_adv = torch.bmm(torch.stack([torch.eye(matrix_size)] * batch_size) - eligible_score, eligible_values)
            if can_wait & torch.any(indicator_wait):
                wait_queues = shift_to_batch(self.queue_linear(embs_wait), ptr_wait)
                wait_queues_mask = ~wait_queues.isinf()
                wait_queues = torch.where(wait_queues_mask, wait_queues, 0)
                wait_keys = shift_to_batch(self.key_linear(embs_wait), ptr_wait)
                wait_keys_mask = ~wait_keys.isinf()
                wait_keys = torch.where(wait_keys_mask, wait_keys, 0)
                wait_values = shift_to_batch(self.value4wait_linear(embs_wait), ptr_wait, padding_value = 0)

                eligible_wait_score_mask = torch.bmm(eligible_queues_mask[indicator_wait].float(), wait_keys_mask.float().permute(0, 2, 1)).bool()
                eligible_wait_score = torch.where(eligible_wait_score_mask, torch.bmm(eligible_queues[indicator_wait], wait_keys.permute(0, 2, 1)), -torch.inf)
                eligible_wait_all_inf_mask = (eligible_wait_score == -torch.inf).all(dim=-1, keepdim=True)
                eligible_wait_score = torch.softmax(eligible_wait_score, dim = -1)
                eligible_wait_score = torch.where(eligible_wait_all_inf_mask, torch.zeros_like(eligible_wait_score), eligible_wait_score)
                eligible_adv[indicator_wait] += eligible_values[indicator_wait] - torch.bmm(eligible_wait_score, wait_values)

                wait_eligible_score_mask = torch.bmm(wait_queues_mask.float(), eligible_keys_mask[indicator_wait].float().permute(0, 2, 1)).bool()
                wait_eligible_score =torch.where(wait_eligible_score_mask, torch.bmm(wait_queues, eligible_keys[indicator_wait].permute(0, 2, 1)), -torch.inf)
                wait_eligible_all_inf_mask = (wait_eligible_score == -torch.inf).all(dim=-1, keepdim=True)
                wait_eligible_score = torch.softmax(wait_eligible_score, dim = -1)
                wait_eligible_score = torch.where(wait_eligible_all_inf_mask, torch.zeros_like(wait_eligible_score), wait_eligible_score)
                wait_adv = wait_values - torch.bmm(wait_eligible_score, eligible_values[indicator_wait])
                

                proc_nodes = self.actor(extract_active_emb(eligible_adv, ptr_proc)).view(-1, 1)
                actor_proc = shift_to_batch(proc_nodes, ptr_proc)
                wait_nodes = self.actor4wait(extract_active_emb(wait_adv, ptr_wait)).view(-1, 1)
                actor_wait = shift_to_batch(wait_nodes, ptr_wait)

                actor_wait_padded = torch.full((actor_proc.size(0), actor_wait.size(1)), fill_value=-float('inf'))
                actor_wait_padded[indicator_wait] = actor_wait
                actor_all = torch.cat([actor_proc, actor_wait_padded], dim=-1)
                action_probs_all = F.softmax(actor_all, dim=-1) if not log_probs else F.log_softmax(actor_all, dim=-1)
                action_probs = torch.zeros(size=(actor_proc.size(0), len_proc+1)) if not log_probs \
                    else torch.full(size=(actor_proc.size(0), len_proc+1), fill_value=-float('inf'))
                action_probs[:, :-1] = action_probs_all[:, :len_proc] 

                wait_vl = action_probs_all[indicator_wait, len_proc:]
                weight = F.softmax(actor_wait, dim=-1)
                action_probs[indicator_wait, -1] = torch.sum(wait_vl * weight, dim=-1) if not log_probs \
                    else torch.logsumexp(wait_vl + weight, dim=-1)
            else:
                proc_nodes = self.actor(extract_active_emb(eligible_adv, ptr_proc)).view(-1, 1)
                actor_proc = shift_to_batch(proc_nodes, ptr_proc)
                if log_probs:
                    return F.log_softmax(actor_proc, dim=-1)
                action_probs = F.softmax(actor_proc, dim=-1)
        else:
            actor_nodes = self.actor(embs_local_global).view(-1,1)
            actor_proc = shift_to_batch(actor_nodes, ptr_proc)
            if can_wait & torch.any(indicator_wait):
                actor_wait = self.actor4wait(embs_wait).view(-1,1)
            
                actor_wait = shift_to_batch(actor_wait, ptr_wait)
                actor_wait_padded = torch.full((actor_proc.size(0), actor_wait.size(1)), fill_value=-float('inf'))
                actor_wait_padded[indicator_wait] = actor_wait
                actor_all = torch.cat([actor_proc, actor_wait_padded], dim=-1)
                action_probs_all = F.softmax(actor_all, dim=-1) if not log_probs else F.log_softmax(actor_all, dim=-1)
                action_probs = torch.zeros(size=(actor_proc.size(0), len_proc+1)) if not log_probs \
                    else torch.full(size=(actor_proc.size(0), len_proc+1), fill_value=-float('inf'))
                action_probs[:, :-1] = action_probs_all[:, :len_proc]
                
                if self.wait_deal == 'max':
                    max_prob_wait_index = torch.argmax(action_probs_all[:, len_proc:], dim=-1)
                    return_index = max_prob_wait_index[indicator_wait]
                    max_prob = action_probs_all[:, len_proc:][torch.arange(action_probs_all.size(0)), max_prob_wait_index]
                    if not log_probs:
                        action_probs[:, -1] = max_prob
                    else:
                        action_probs[:, -1] = max_prob
                elif self.wait_deal == 'softmax':
                    wait_vl = action_probs_all[indicator_wait, len_proc:]
                    weight = F.softmax(shift_to_batch(self.actor(embs_wait).view(-1,1), ptr_wait), dim=-1)
                    action_probs[indicator_wait, -1] = torch.sum(wait_vl * weight, dim=-1) if not log_probs \
                        else torch.logsumexp(wait_vl + weight, dim=-1)
                    
                elif self.wait_deal == 'mean':
                    wait_size = (ptr_wait[1:] - ptr_wait[:-1])
                    action_probs[indicator_wait, -1] = torch.sum(action_probs_all[indicator_wait, len_proc:], dim=-1)/wait_size if not log_probs \
                        else torch.logsumexp(action_probs_all[indicator_wait, len_proc:], dim=-1) - torch.log(wait_size)
                elif self.wait_deal == 'sum':
                    action_probs[indicator_wait, -1] = torch.sum(action_probs_all[indicator_wait, len_proc:], dim=-1) if not log_probs \
                        else torch.logsumexp(action_probs_all[indicator_wait, len_proc:], dim=-1)
                elif self.wait_deal == 'weighted':
                    with torch.no_grad():
                        wait_in_actor = F.softmax(self.actor(embs_wait).view(-1,1), dim= -1)
                    wait_in_actor_padded = torch.full((actor_proc.size(0), wait_in_actor.size(1)), fill_value=-float('inf'))
                    wait_in_actor_padded[indicator_wait] = wait_in_actor
                    action_probs[indicator_wait, -1] = torch.sum(action_probs_all[indicator_wait, len_proc:] * wait_in_actor_padded, dim=-1) if not log_probs \
                        else torch.logsumexp(action_probs_all[indicator_wait, len_proc:] + wait_in_actor, dim=-1)
            else:
                if log_probs:
                    return F.log_softmax(actor_proc, dim=-1)
                action_probs = F.softmax(actor_proc, dim=-1)
        

        if torch.isnan(action_probs).any():
            raise ValueError("Nan in action_probs!")
        if return_max_wait_idxes:
            return action_probs, return_index
        return action_probs
    
class DRLPolicy(Policy):
    """
    Deep Reinforcement Learning Policy class.

    This class represents a policy for deep reinforcement learning. It extends the base Policy class.

    More Attributes:
        critic (Critic): The critic network.
        wait_flag (bool): A flag indicating whether to wait.
        priority_coef (float): The priority coefficient.

    Methods:
        __init__(self, config): Initializes the DRLPolicy object.
        act(self, state, memory, flag_sample=True, flag_train=False): Selects an action based on the given state.
        evaluate(self, graph, eligible, future_eligible, opes_appertain, actions, batch_idxes, waits=None): Evaluates the action log probabilities, Q values, and entropy for a given state.
    """
    def __init__(self, config):
        super(DRLPolicy, self).__init__(config)
        self.critic = Critic(config)
        self.wait_flag = config.wait_flag
        self.priority_coef = config.priority_coef
        
    def act(
            self,
            state: EnvState,
            memory: MemoryRL,
            flag_sample: bool = True,
            flag_train: bool = False
            ):
        """
        Selects an action based on the given state.

        Args:
            state: The input state.
            memory: The memory object.
            flag_sample: A flag indicating whether to sample actions (default: True).
            flag_train: A flag indicating whether to train the agent (default: False).

        Returns:
            action: The selected action.
        """
        with torch.no_grad():
            
            # Get the eligible pairs
            batch_idxes = state.batch_idxes
            
            eligible = state.eligible_pairs
            if (~(eligible)).all():
                raise ValueError("No eligible pairs")
            

            
            _, embs_local_global, ptr_proc, embs_wait, ptr_wait, indicator_wait \
                    = self.get_embedding(state.graph, eligible, state.future_eligible_pairs, state.opes_appertain_batch.long(), batch_idxes)
                    
                
            if self.wait_deal == 'max':
                action_probs, wait_max_idxes = self.embs_to_probs(embs_local_global, ptr_proc, embs_wait, ptr_wait, \
                    indicator_wait=indicator_wait, return_max_wait_idxes=True)
            elif self.wait_deal == 'mean' or self.wait_deal == 'softmax':
                action_probs = self.embs_to_probs(embs_local_global, ptr_proc, embs_wait, ptr_wait, \
                    indicator_wait=indicator_wait)
            elif self.wait_deal == 'flexible':
                action_probs = self.embs_to_probs(embs_local_global, ptr_proc, embs_wait, ptr_wait, \
                    indicator_wait=indicator_wait, eligible = eligible, future_eligible = state.future_eligible_pairs)
            
            if torch.isnan(action_probs).any():
                raise ValueError("Nan in action_probs!")
            
            # DRL-S, sampling actions following \pi
            if flag_sample:
                dist = Categorical(action_probs)
                action_indexes = dist.sample()
            # DRL-G, greedily picking actions with the maximum probability
            else:
                action_indexes = action_probs.argmax(dim=1)

            job_ope_relation = state.opes_appertain_batch[batch_idxes]
            action = torch.zeros(3, batch_idxes.size(0), dtype=torch.long)

            wait_idx = torch.nonzero(indicator_wait).squeeze(-1)
            padded_wait_idx = torch.scatter(torch.full((action_indexes.size(0),), -1, dtype=torch.long), 
                                            0, wait_idx, torch.arange(wait_idx.size(0), device=self.device))



            # Choose the action And get the action embedding
            for i in range(action_indexes.size(0)):
                eligible_pair = eligible[i].nonzero().t()
                if action_indexes[i] >= eligible_pair.size(1):
                    action[:, i] = int(-1)
                    index = padded_wait_idx[i]
                    assert index != -1

                else:
                    action[0, i] = eligible_pair[0, action_indexes[i]].long()
                    action[1, i] = eligible_pair[1, action_indexes[i]].long()
                    action[2, i] = job_ope_relation[i, action[0, i]].long()


            # Update the memory
            if flag_train:
                assert flag_sample, 'Only support sampling actions during training.'
                batchsize = state.opes_appertain_batch.size(0)
                log_probs = dist.log_prob(action_indexes)
                
                # Padding the Date we need to store
                padded_indices = torch.zeros(batchsize, device=self.device, dtype=torch.bool)
                padded_indices[state.batch_idxes] = 1
                padded_eligible = torch.zeros((batchsize, *eligible.size()[1:]), device=self.device, dtype=eligible.dtype)
                padded_eligible[state.batch_idxes] = eligible

                padded_action_indexes = torch.zeros((batchsize,), dtype=torch.long, device=self.device)
                padded_action_indexes[state.batch_idxes] = action_indexes
                padded_log_probs = torch.zeros((batchsize,), dtype=torch.float, device=self.device)
                padded_log_probs[state.batch_idxes] = log_probs
                padded_waits = torch.zeros((batchsize,), dtype=torch.long, device=self.device)
                padded_waits[state.batch_idxes] = torch.where(action[0] == -1, 1 ,0)
                padded_future_eligible = torch.zeros((batchsize, *state.future_eligible_pairs.size()[1:]), device=self.device, dtype=state.future_eligible_pairs.dtype)
                padded_future_eligible[state.batch_idxes] = state.future_eligible_pairs
                
                memory.eligible.append(padded_eligible)
                memory.graphs.append(copy.deepcopy(state.graph.data.to_data_list()))
                memory.batch_idxes.append(copy.deepcopy(state.batch_idxes))
                memory.actives.append(padded_indices)
                memory.logprobs.append(padded_log_probs)
                memory.action_indexes.append(padded_action_indexes)
                memory.waits.append(padded_waits)
                memory.future_eligible.append(padded_future_eligible)
                memory.opes_appertain.append(state.opes_appertain_batch)

        
        return action
    
    def evaluate(self, graph, eligible, future_eligible, opes_appertain, actions, batch_idxes, waits = None):
        """
        Evaluates the action log probabilities, Q values, and entropy for a given state.

        Args:
            graph: The input graph.
            eligible: The eligibility mask.
            actions: The selected actions.
            batch_idxes: The batch indices.
            waits: The wait actions.

        Returns:
            action_logprobs: The action log probabilities.
            Q_values: The Q values.
            dist_entropys: The entropy values.
        """
        embs_global, embs_local_global, ptr, embs_wait, ptr_wait, indicator_wait\
            = self.get_embedding(graph, eligible, future_eligible, opes_appertain, batch_idxes)
            

        action_probs = self.embs_to_probs(embs_local_global, ptr, embs_wait, ptr_wait, indicator_wait)
        dist = Categorical(action_probs.squeeze())
        wait_indice = action_probs.size(-1) - 1
        actions = torch.where(waits == 1, wait_indice , actions)

        action_logprobs = dist.log_prob(actions)
        dist_entropys = dist.entropy()
        Q_values = self.critic(embs_global)
        
        return action_logprobs, Q_values, dist_entropys

class ExpertPolicy(Policy):
    """
        Using for BC inference
    """
    def __init__(self, config):
        super(ExpertPolicy, self).__init__(config)
        self.wait_flag = True

    def act(
            self, 
            state: EnvState, 
            memory: MemoryIL, 
            flag_sample: bool = True, 
            ):
        """
        Selects an action based on the given state.

        Args:
            state: The input state.
            memory: The memory object.
            flag_sample: A flag indicating whether to sample actions (default: True).
            flag_train: A flag indicating whether to train the agent (default: True).

        Returns:
            action: The selected action.
        """
        with torch.no_grad():
            
            # Get the eligible pairs
            batch_idxes = state.batch_idxes
            
            eligible = state.eligible_pairs
            if (~(eligible)).all():
                raise ValueError("No eligible pairs")
            
            _, embs_local_global, ptr_proc, embs_wait, ptr_wait, indicator_wait \
                    = self.get_embedding(state.graph, eligible, state.future_eligible_pairs, state.opes_appertain_batch.long(), batch_idxes)
                        

            action_probs = self.embs_to_probs(embs_local_global, ptr_proc, embs_wait, ptr_wait, \
                indicator_wait=indicator_wait)
            
            if torch.isnan(action_probs).any():
                raise ValueError("Nan in action_probs!")
            if not self.wait_flag:
                action_probs = action_probs[:, :-1] #FIXME
            
            # DRL-S, sampling actions following \pi
            if flag_sample:
                dist = Categorical(action_probs)
                action_indexes = dist.sample()
            # DRL-G, greedily picking actions with the maximum probability
            else:
                action_indexes = action_probs.argmax(dim=1)

            job_ope_relation = state.opes_appertain_batch[batch_idxes]
            action = torch.zeros(3, batch_idxes.size(0), dtype=torch.long)


            wait_idx = torch.nonzero(indicator_wait).squeeze(-1)
            padded_wait_idx = torch.scatter(torch.full((action_indexes.size(0),), -1, dtype=torch.long), 
                                            0, wait_idx, torch.arange(wait_idx.size(0), device=self.device))



            # Choose the action And get the action embedding
            for i in range(action_indexes.size(0)):
                eligible_pair = eligible[i].nonzero().t()
                if action_indexes[i] >= eligible_pair.size(1):
                    action[:, i] = int(-1)
                    index = padded_wait_idx[i]
                    assert index != -1
                    
                else:
                    action[0, i] = eligible_pair[0, action_indexes[i]].long()
                    action[1, i] = eligible_pair[1, action_indexes[i]].long()
                    action[2, i] = job_ope_relation[i, action[0, i]].long()


        return action
