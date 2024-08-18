import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_scatter import scatter
from torch_geometric.nn import GraphNorm
def get_indices(hetero_data:HeteroData, batch_idx:torch.Tensor, local_indices:torch.Tensor):
    """Caculate the indices in Batch Graph according to the batch index and local indices.

    Args:
        hetero_data (HeteroData): a batch of graph data.
        batch_idx (Tensor): Indicating the index of the batch.
        local_indices (Tensor): Indicating the local index in the batch.

    Returns:
        opes_indice, mas_indice: The indices of the opes and mas nodes in the batch graph.
    """
    ope_indice = hetero_data['opes'].ptr[batch_idx].unsqueeze(-1)+local_indices[:, 0, :]
    mas_indice = hetero_data['mas'].ptr[batch_idx].unsqueeze(-1)+local_indices[:, 1, :]
    job_indice = hetero_data['jobs'].ptr[batch_idx].unsqueeze(-1)+local_indices[:, 2, :]
    return ope_indice, mas_indice, job_indice

def get_indice_ptr_eligible(hetero_data:HeteroData, eligible_nodes:torch.Tensor, opes_appertain:torch.Tensor, batch_idx:torch.Tensor):
    eligible_ptr = torch.zeros(eligible_nodes.size(0) + 1, dtype=torch.long)
    eligible_choice = torch.Tensor()
    job_choice = torch.Tensor()

    graph_ptr = torch.cat([hetero_data['opes'].ptr[batch_idx].unsqueeze(-1), hetero_data['mas'].ptr[batch_idx].unsqueeze(-1)], dim=-1)
    for i in range(eligible_nodes.size(0)):
        sparse_edge = eligible_nodes[i].nonzero().t() 
        if i == 0:
            eligible_choice = sparse_edge + graph_ptr[i].unsqueeze(-1).expand_as(sparse_edge)
            eligible_ptr[i] = 0
            eligible_ptr[i+1] = sparse_edge.size(-1)
            job_choice = opes_appertain[i].gather(0, sparse_edge[0].long())
        else:
            eligible_choice = torch.cat([eligible_choice, sparse_edge + graph_ptr[i].unsqueeze(-1).expand_as(sparse_edge)], dim=-1)
            job_choice = torch.cat([job_choice, opes_appertain[i].gather(0, sparse_edge[0].long())], dim=-1).long()
            eligible_ptr[i+1] = eligible_ptr[i] + sparse_edge.size(-1)
    opes_indice, mas_indice = eligible_choice[0], eligible_choice[1]
    return opes_indice, mas_indice, job_choice, eligible_ptr
    
def get_graph_embs_to_pair(graph_embedding:HeteroData, eligible_nodes:torch.Tensor, opes_appertain:torch.Tensor, batch_idx:torch.Tensor, return_indices=False):
    """Get the embedding of each node being eligible.
    Args:
        graph_embedding: The graph embedding model.
        eligible_nodes: The eligible nodes.
    Returns:
        eligible_embedding: The embedding of the eligible nodes.
        eligible_ptr: The pointer of the eligible nodes in the batch graph.
        Shape: [num_eligible_pairs, 2, hidden_dim], [batch_size]
    """
    opes_indice, mas_indice, job_choice, eligible_ptr = get_indice_ptr_eligible(graph_embedding, eligible_nodes, opes_appertain, batch_idx)
    ope_eligible_embedding = graph_embedding['opes'].x[opes_indice]
    mas_eligible_embedding = graph_embedding['mas'].x[mas_indice]
    job_eligible_embedding = graph_embedding['jobs'].x[job_choice.long()]

    if return_indices:
        return torch.cat([ope_eligible_embedding, mas_eligible_embedding, job_eligible_embedding], dim=-1), eligible_ptr, opes_indice, mas_indice, job_choice
    return torch.cat([ope_eligible_embedding, mas_eligible_embedding, job_eligible_embedding], dim=-1), eligible_ptr

def global_node_pooling(node_embeddings: torch.Tensor, ptr: torch.Tensor, pool_type='mean', mask=None):
    """Get the global embedding of the graph using pooling.
    Args:
        node_embedding: The embedding of the nodes.
        ptr: The pointer of the nodes in the batch graph.
        pool_type: The pooling type.
        mask: The indices of the nodes to be pooled.
    Returns:
        global_node_embedding: Shape: [batch_size, hidden_dim], The global node embeddings of the graph.
    """
    # Create batch index tensor
    batch = torch.arange(len(ptr) - 1).repeat_interleave(ptr[1:] - ptr[:-1])
    node_embeddings = node_embeddings if mask is None else node_embeddings[mask]
    batch = batch if mask is None else batch[mask]
    # Apply scatter operation for pooling
    if pool_type == 'mean':
        global_node_embedding = scatter(node_embeddings, batch, dim=0, reduce='mean')
    elif pool_type == 'max':
        global_node_embedding = scatter(node_embeddings, batch, dim=0, reduce='max')
    elif pool_type == 'sum':
        global_node_embedding = scatter(node_embeddings, batch, dim=0, reduce='sum')
    else:
        raise ValueError(f"Unsupported pool_type: {pool_type}")


    global_node_padding = torch.zeros(ptr.size(0) - 1, global_node_embedding.size(-1))
    if len(batch) != 0:
        global_node_padding[0:batch.max()+1] = global_node_embedding
    return global_node_padding

def expand_global_embedding(global_embedding, ptr):
    """Expand the global embedding according to ptr.
    Args:
        global_embedding: The global embedding tensor.
        ptr: The batch pointers indicating the start of each graph in the batch.
    Returns:
        expanded_global_embedding: The expanded global embedding tensor.
    """
    # Create an index tensor to repeat global embeddings
    num_nodes = ptr[1:] - ptr[:-1]
    expanded_global_embedding = global_embedding.repeat_interleave(num_nodes, dim=0)
    
    return expanded_global_embedding

def cat_local_global_embedding(local_embedding: torch.Tensor, global_embedding: torch.Tensor, ptr: torch.Tensor):
    """Concatenate the local and global embedding.
    Args:
        local_embedding: The local embedding.
        global_embedding: The global embedding.
        ptr: The pointer of the nodes in the batch graph.
    Returns:
        local_global_embedding: Shape: [num_nodes, 2*hidden_dim], The concatenated local and global embedding.
    """
    return torch.cat(
        [local_embedding, expand_global_embedding(global_embedding, ptr)], dim=-1
    )
    
def shift_to_batch(output:torch.Tensor, ptr:torch.Tensor, wait_output = None, padding_value = -float('inf')):
    """
        From the output of the model, shift to a batch tensor.
    """
    batch_size = len(ptr) - 1
    segments = [output[ptr[i]:ptr[i+1]].squeeze(-1) for i in range(batch_size)]
    padded_tensor = nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value = padding_value)
    if wait_output is not None:
        padded_tensor = torch.cat([padded_tensor, wait_output.view(-1,1)], dim=-1)
    return padded_tensor

def extract_active_emb(tensor:torch.Tensor, ptr:torch.Tensor):
    batch_size = len(ptr) - 1
    new_tensor = []
    for i in range(batch_size):
        new_tensor.append(tensor[i, 0:(ptr[i+1] - ptr[i])])
    return torch.cat(new_tensor, dim = 0)


def global_pooling(graph_embedding: HeteroData, pool_type='mean'):
    """Get the global embedding of the graph using pooling.
    Note: The isolated nodes are removed before pooling.
    Args:
        graph_embedding: The graph embedding.
        ptr: The pointer of the nodes in the batch graph.
        pool_type: The pooling type.
    Returns:
        global_embedding: Shape: [batch_size, hidden_dim], The global embedding of the graph.
    """

    ope_remain_indices = non_isolated_node_indices(graph_embedding['mas', 'proc', 'opes'].edge_index, 1)
    mas_remain_indices = non_isolated_node_indices(graph_embedding['mas', 'proc', 'opes'].edge_index, 0)
    job_remain_indices = non_isolated_node_indices(graph_embedding['combs', 'belong_job', 'jobs'].edge_index, 1)
    opes_global_embedding = global_node_pooling(graph_embedding['opes'].x, graph_embedding['opes'].ptr, 
                                                pool_type, ope_remain_indices)
    mas_global_embedding = global_node_pooling(graph_embedding['mas'].x, graph_embedding['mas'].ptr, 
                                               pool_type, mas_remain_indices)
    job_global_embedding = global_node_pooling(graph_embedding['jobs'].x, graph_embedding['jobs'].ptr, 
                                               pool_type, job_remain_indices)
    return torch.cat([opes_global_embedding, mas_global_embedding, job_global_embedding], dim=-1)

def non_isolated_node_indices(edge_index: torch.Tensor, end_index: torch.Tensor):
    """Get the indices of the non-isolated nodes.
    Args:
        edge_index: The edge index tensor.
        end_index: 0 fisrt, 1 second
    Returns:
        non_isolated_indices: The indices of the non-isolated nodes.
    """
    return torch.unique(edge_index[end_index])


def normalize_hetero_data(data: HeteroData):
    """
    Normalize the features of the nodes and edge attributes in the given HeteroData object.

    Args:
        data (HeteroData): The input HeteroData object containing node and edge attributes.

    Returns:
        HeteroData: The normalized HeteroData object with updated node and edge attributes.
    """

    for node_type in data.node_types:
        if 'x' in data[node_type]:
            x = data[node_type].x
            data[node_type].x = GraphNorm(x.size(-1))(x, data[node_type].batch)
    for edge_type in data.edge_types:
        if 'edge_attr' in data[edge_type]:
            edge_attr = data[edge_type].edge_attr
            edge_batch = torch.repeat_interleave(torch.arange(data._slice_dict[edge_type]['edge_index'].size(0) - 1), \
                data._slice_dict[edge_type]['edge_index'][1:] - data._slice_dict[edge_type]['edge_index'][:-1])
            if edge_batch.numel() != 0:
                data[edge_type].edge_attr = GraphNorm(edge_attr.size(-1))(edge_attr, edge_batch)
    return data

def get_remain_edges_per_edge_type(graph:HeteroData, edge_type:tuple, remain_opes_flatten:torch.Tensor, deal_start = True, deal_end = True):
    edge_index = graph[edge_type].edge_index
    if deal_start and deal_end:
        mask = torch.isin(edge_index[0], remain_opes_flatten) & torch.isin(edge_index[1], remain_opes_flatten)
    elif deal_start:
        mask = torch.isin(edge_index[0], remain_opes_flatten)
    elif deal_end:
        mask = torch.isin(edge_index[1], remain_opes_flatten)
    else:
        return
    return mask
    
def get_remain_edges_per_node_type(graph: HeteroData, node_type:str, remain_opes_flatten:torch.Tensor):
    remain_dict = {}
    for edge_type in graph.edge_types:
        is_opes_start = edge_type[0] == node_type
        is_opes_end = edge_type[2] == node_type
        remain = get_remain_edges_per_edge_type(
            graph,
            edge_type,
            remain_opes_flatten,
            deal_start=is_opes_start,
            deal_end=is_opes_end
        )
        if remain is not None and ~remain.all():
            remain_dict[edge_type] = remain
    return remain_dict