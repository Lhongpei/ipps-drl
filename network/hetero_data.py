from torch_geometric.data import HeteroData, Batch
from network.utils import get_remain_edges_per_node_type
from itertools import product
import torch
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('...')
from env.load_data import load_ipps
import torch
from utils.utils import flatten_padded_tensor   
from torch_scatter import scatter_max
from torch_scatter import scatter
import copy
class Graph_Batch:
    '''Represents a batch of heterogeneous graphs for each batch of problems.

    The structure of the graph is:
    - Node types: 'opes', 'mas', 'combs', 'jobs'
    - Edge types: 
        - 'pre': opes -> opes
        - 'proc': opes <-> mas
        - 'belong_comb': combs <-> opes
        - 'belong_job': combs <-> job

    Args:
        graph_list (list, optional): List of data objects representing individual graphs. Defaults to an empty list.

    Attributes:
        batch_size (int): Number of graphs in the batch.
        data (HeteroData): Heterogeneous graph data.
        batch_iter (torch.Tensor): Iterator for batch processing.
        opes_ptr (torch.Tensor): Pointers to the start of each operation subgraph within the batch.
        mas_ptr (torch.Tensor): Pointers to the start of each machine subgraph within the batch.
        ope_num (torch.Tensor): Number of operations for each subgraph in the batch.
        mas_num (torch.Tensor): Number of machines for each subgraph in the batch.

    Methods:
        load_features(load_data): Load features for a single problem.
        create_batch(graph_list): Convert a list of data objects into a single batched graph.
        get_opes_indices(batch_idx, local_indices): Calculate the global indices for operations based on batch index and local indices.
        get_mas_indices(batch_idx, local_indices): Calculate the global indices for machines based on batch index and local indices.
        update_features(batch_idxes, actions, end_time, ready_time, time, machine, ope_req_num_batch, ope_step_batch): Update the features for operations and machines based on the given actions and times.
    '''
    
    def __init__(self, graph_list=[]):
        # Initialize with an empty list by default, set the batch size based on the list
        self.batch_size = len(graph_list)
        self.data = HeteroData()
        self.batch_iter = torch.arange(self.batch_size) if graph_list else None
        #pointers to the start of each operation and machine subgraphs within the batch
        self.opes_ptr = None
        self.mas_ptr = None
        self.combs_ptr = None
        self.jobs_ptr = None
        #ope/mas numbers of each subgraph
        self.ope_num = None
        self.mas_num = None
        self.combs_num = None
        self.job_num = None
        # If there are pre-existing graphs in the batch, construct the heterogeneous graph for that batch
        if graph_list:
            # Convert a list of data objects into a single batched graph
            self.create_batch(graph_list)

    def load_features(self, load_data)->HeteroData:
        '''Load features for a single problem.

        Args:
            load_data: Data for a single problem.

        Returns:
            HeteroData: Heterogeneous graph data with loaded features.
        '''
        matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc,\
            start, num_opes, ope_req_num, ope_eligible, \
                combs_id_batch, combs_batch = self._extract_from_load_data(load_data)
        ope_num=ope_req_num.size(0)
        job_num = combs_id_batch.size(0)
        combs_num = combs_batch.size(0)
        num_mas = matrix_proc_time.size(1)
        end = start + num_opes - 1
        pro_matrix_ope_ma_adj = copy.deepcopy(matrix_ope_ma_adj)
        pro_matrix_ope_ma_adj[end] = 0
        data = HeteroData()# Initialize a HeteroData object for storing graph data

        '''operation features with shape [ope_num,5]
        '''
        feat_opes= torch.zeros(size=(ope_num,5))# Create operation features with shape [ope_num, 5]
        opes_status=torch.zeros_like(ope_req_num)
        feat_opes[:,0]=ope_req_num
        feat_opes[:,1]=opes_status
        feat_opes[start, 1] = 1
        feat_opes[:,2]=ope_eligible
        data['opes'].x =feat_opes

        '''machine features with shape[mas_num,6]
        '''

        feat_mas= torch.zeros(size=(num_mas,6)) # Initialize machine features with shape [mas_num, 6]
        feat_mas[:,0]=torch.count_nonzero(pro_matrix_ope_ma_adj.t(), dim=1)
        data['mas'].x = feat_mas
        '''
            Combinations feature with shape[cobms_num,2]
            Job feature with shape[job_num,1]
        '''
        data['combs'].x = torch.ones(size=(combs_num,2))
        data['jobs'].x = torch.ones(size=(job_num,1))
        
        '''
            Edge features and index.
        '''
        data['opes', 'pre', 'opes'].edge_index = (matrix_pre_proc.t()).nonzero().t()
        data['mas', 'proc', 'opes'].edge_index = pro_matrix_ope_ma_adj.t().nonzero().t()
        data['opes', 'proc_rev', 'mas'].edge_index = data['mas', 'proc', 'opes'].edge_index[[1,0]]
        data['opes', 'belong_comb', 'combs'].edge_index = combs_batch.t().nonzero().t()
        data['combs', 'belong_comb_rev', 'opes'].edge_index = data['opes', 'belong_comb', 'combs'].edge_index[[1,0]]
        data['combs', 'belong_job', 'jobs'].edge_index = combs_id_batch.t().nonzero().t()
        data['jobs', 'belong_job_rev', 'combs'].edge_index = data['combs', 'belong_job', 'jobs'].edge_index[[1,0]]
        
        machine_indices, operation_indices = data['mas', 'proc', 'opes'].edge_index
        processing_times = matrix_proc_time.t()
        data['mas', 'proc', 'opes'].edge_attr = processing_times[machine_indices, operation_indices].view(-1,1)
        data['opes', 'proc_rev', 'mas'].edge_attr = data['mas', 'proc', 'opes'].edge_attr
        return data
    
    def _extract_from_load_data(self, load_data):
        matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, \
        _, start, num_opes, _, \
        _, ope_req_num, ope_eligible, combs_id_batch, combs_batch, _ = load_data
        return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, start, num_opes, ope_req_num, ope_eligible, combs_id_batch, combs_batch
    
    def cat_load_features(self, load_data, old_data:HeteroData):

        new_data = self.load_features(load_data)
        data = HeteroData()
        data['opes'].x = torch.cat([old_data['opes'].x, new_data['opes'].x], dim=0)
        data['combs'].x = torch.cat([old_data['combs'].x, new_data['combs'].x], dim=0)
        data['jobs'].x = torch.cat([old_data['jobs'].x, new_data['jobs'].x], dim=0)
        feat_mas= torch.zeros(size=(max(new_data['mas'].x.size(0), old_data['mas'].x.size(0)),6)) # Initialize machine features with shape [mas_num, 6]
        feat_mas[:old_data['mas'].x.size(0)] = old_data['mas'].x
        feat_mas[:new_data['mas'].x.size(0)] += new_data['mas'].x
        data['mas'].x = feat_mas
        for edge_type in old_data.edge_types:
            start_type, end_type = edge_type[0], edge_type[2]
            is_start_mas = start_type == 'mas'
            is_end_mas = end_type == 'mas'
            new_data[edge_type].edge_index[0] = new_data[edge_type].edge_index[0] if is_start_mas else new_data[edge_type].edge_index[0] + old_data[start_type].x.size(0)
            new_data[edge_type].edge_index[1] = new_data[edge_type].edge_index[1] if is_end_mas else new_data[edge_type].edge_index[1] + old_data[end_type].x.size(0)
            data[edge_type].edge_index = torch.cat([old_data[edge_type].edge_index, new_data[edge_type].edge_index], dim=1)
            if getattr(old_data[edge_type], 'edge_attr', None) is not None:
                data[edge_type].edge_attr = torch.cat([old_data[edge_type].edge_attr, new_data[edge_type].edge_attr], dim=0)
        return data
            
    
    def create_batch(self,graph_list):
        '''Convert a list of data objects into a single batched graph.

        Args:
            graph_list (list): List of data objects representing individual graphs.
        '''

        self.data = Batch.from_data_list(graph_list)
        self.batch_size = len(graph_list)

        # Iterator for batch processing
        self.batch_iter = torch.arange(self.batch_size)
        
        # Store pointers to the start of each operation and machine subgraphs within the batch
        self.batch_iter = torch.arange(self.batch_size)
        self.opes_ptr = self.data['opes'].ptr[:-1]
        self.mas_ptr = self.data['mas'].ptr[:-1]
        self.combs_ptr = self.data['combs'].ptr[:-1]
        self.jobs_ptr = self.data['jobs'].ptr[:-1]
        #Store number of operations and machines for each problem
        self.ope_num=torch.bincount(self.data['opes'].batch, minlength=self.batch_size)
        self.mas_num=torch.bincount(self.data['mas'].batch, minlength=self.batch_size)

    def init_from_graph_list(self, 
                             graph_list: list, 
                             combs_time_batch: torch.Tensor, 
                             remain_opes: torch.Tensor,
                             job_estimate_time: torch.Tensor
                             ):
        self.create_batch(graph_list)
        self.data['combs'].x[:, 0] = flatten_padded_tensor(self.data['combs'].ptr, combs_time_batch)
        edge_min_combs_in_job =  self.data['combs'].x[ self.data['combs', 'belong_job', 'jobs'].edge_index[0], 0].squeeze(-1)
        min_combs_in_job = scatter(edge_min_combs_in_job,  self.data['combs', 'belong_job', 'jobs'].edge_index[1], dim=0, reduce="min")[self.data['combs', 'belong_job', 'jobs'].edge_index[1]]
        min_combs_in_job_to_combs = scatter(min_combs_in_job, self.data['combs', 'belong_job', 'jobs'].edge_index[0], dim=0, reduce="min")
        self.data['combs'].x[:min_combs_in_job_to_combs.size(-1), 1] = min_combs_in_job_to_combs/ (self.data['combs'].x[:min_combs_in_job_to_combs.size(-1), 0] + 1e-5)
        self.data['jobs'].x[:, 0] = flatten_padded_tensor(self.data['jobs'].ptr, job_estimate_time/(job_estimate_time.max() + 1e-5))
        self.update_edge_sub_graph(remain_opes)
        
    def get_opes_indices(self, batch_idx,local_indices):
        '''Calculate the global indices for operations based on batch index and local indices.

        Args:
            batch_idx (int): Batch index.
            local_indices (torch.Tensor): Local indices within the batch.

        Returns:
            torch.Tensor: Global indices for operations.
        '''
        return self.opes_ptr[batch_idx]+local_indices
    
    def get_mas_indices(self, batch_idx,local_indices):
        '''Calculate the global indices for machines based on batch index and local indices.

        Args:
            batch_idx (int): Batch index.
            local_indices (torch.Tensor): Local indices within the batch.

        Returns:
            torch.Tensor: Global indices for machines.
        '''
        return self.mas_ptr[batch_idx]+local_indices
    
        
    def get_edge_sub_graph(self, remain_opes, remain_combs = None):
        """
            old version of updaing edge sub graph by spliting the graph into data_list and concating list of graph, which is not efficient.
            Deprecated.
        """
        data_list = self.data.to_data_list()
        for i,graph in enumerate(data_list):
            remain_opes_flatten = remain_opes[i].nonzero().squeeze(-1)
            ope_remain_dict = get_remain_edges_per_node_type(graph, 'opes', remain_opes)
            if remain_combs is not None:
                remain_combs_flatten = remain_combs[i].nonzero().squeeze(-1)
                combs_remain_dict = get_remain_edges_per_node_type(graph, 'combs', remain_combs_flatten)
                ope_remain_dict.update(combs_remain_dict)
            data_list[i] = graph.edge_subgraph(ope_remain_dict)
        self.data = Batch.from_data_list(data_list)
        
    def update_edge_sub_graph(self, remain_opes, remain_combs=None):
        """
        Update the edge subgraph based on the remaining operations and combinations.

        Args:
            remain_opes (Tensor): A tensor containing the indices of the remaining operations.
            remain_combs (Tensor, optional): A tensor containing the indices of the remaining combinations. Defaults to None.
        """
        remain_opes_flatten = flatten_padded_tensor(self.data['opes'].ptr, remain_opes).nonzero().squeeze(-1)
        self.update_remain_edges_per_node_type('opes', remain_opes_flatten, self.data['opes'].ptr)
        if remain_combs is not None:
            remain_combs_flatten = flatten_padded_tensor(self.data['combs'].ptr, remain_combs).nonzero().squeeze(-1)
            self.update_remain_edges_per_node_type('combs', remain_combs_flatten, self.data['combs'].ptr)
        
    def update_remain_edges_per_edge_type(self, edge_type:tuple, ptr, remain_opes_flatten:torch.Tensor, deal_start=True, deal_end=True):
        """
        Update the remaining edges per edge type based on the specified conditions.
        Avoiding for loop to update the pointer of edge_index and edge_attr to improve efficiency.

        Args:
            edge_type (tuple): The edge type to update.
            ptr: The pointer tensor.
            remain_opes_flatten (torch.Tensor): The tensor containing the remaining edges.
            deal_start (bool, optional): Whether to deal with the start index of the edge. Defaults to True.
            deal_end (bool, optional): Whether to deal with the end index of the edge. Defaults to True.

        Returns:
            None
        """
        edge_index = self.data[edge_type].edge_index
        if deal_start and deal_end:
            target_index = edge_index[0]
            mask = torch.isin(edge_index[0], remain_opes_flatten) & torch.isin(edge_index[1], remain_opes_flatten)
        elif deal_start:
            target_index = edge_index[0]
            mask = torch.isin(edge_index[0], remain_opes_flatten)
        elif deal_end:
            target_index = edge_index[1]
            mask = torch.isin(edge_index[1], remain_opes_flatten)
        else:
            return
        if (~mask).sum() == 0:
            return
        matching_indices = torch.where(~mask)[0]
        ptr_pad = torch.repeat_interleave(torch.arange(ptr.size(0) - 1), ptr[1:] - ptr[:-1])
        edge_belong_batch = ptr_pad[target_index[matching_indices]]
        del_indice, delete_num = torch.unique(edge_belong_batch, return_counts=True)
        padded_delete_num = torch.zeros(ptr.size(0) - 1, dtype=delete_num.dtype)
        padded_delete_num[del_indice] = delete_num
        self.data[edge_type].edge_index = edge_index[:, mask]

        self.data._slice_dict[edge_type]['edge_index'][1:] -= torch.cumsum(padded_delete_num, dim=0)
        assert (self.data._slice_dict[edge_type]['edge_index'] >= 0).all() and (self.data._slice_dict[edge_type]['edge_index'][-1] == self.data[edge_type].edge_index.size(1))
        if getattr(self.data[edge_type], 'edge_attr', None) is not None:
            self.data[edge_type].edge_attr = self.data[edge_type].edge_attr[mask]
            self.data._slice_dict[edge_type]['edge_attr'][1:] -= torch.cumsum(padded_delete_num, dim=0)
            assert (self.data._slice_dict[edge_type]['edge_attr'] >= 0).all() and (self.data._slice_dict[edge_type]['edge_attr'][-1] == self.data[edge_type].edge_attr.size(0))
        return
    
    def update_remain_edges_per_node_type(self, node_type: str, remain_opes_flatten: torch.Tensor, ptr: torch.Tensor):
        """
        Update the remaining edges per node type.
        if edge links same node type, we will remain the edge if the start and end node are both in the remain_opes_flatten.

        Args:
            node_type (str): The type of the node.
            remain_opes_flatten (torch.Tensor): The flattened tensor of remaining edges.
            ptr (torch.Tensor): The tensor representing the pointer.

        Returns:
            None
        """
        for edge_type in self.data.edge_types:
            is_opes_start = edge_type[0] == node_type
            is_opes_end = edge_type[2] == node_type
            self.update_remain_edges_per_edge_type(
                edge_type,
                ptr,
                remain_opes_flatten,
                deal_start=is_opes_start,
                deal_end=is_opes_end
            )
    
    def update_features(self, batch_idxes, actions, end_time, ready_time, time, machine, ope_req_num_batch, combs_time_batch, combs_id_batch, remain_opes, job_estimate_time):
        '''Update the features for operations and machines based on the given actions and times.

        Args:
            batch_idxes (torch.Tensor): Indices of the batches being updated.
            actions (torch.Tensor): Actions taken, where each row corresponds to operations, machines, and jobs respectively.
            end_time (torch.Tensor): End times for the operations.
            ready_time (torch.Tensor): Ready times for the operations to start.
            time (torch.Tensor): Current simulation times for each batch.
            machine (torch.Tensor): Machine data, including availability and utilization.
            ope_req_num_batch (torch.Tensor): Operation requirements per batch.
            ope_step_batch (torch.Tensor): Current steps of operations being processed in each batch.
        '''

        # Extract actions for operations, machines, and jobs
        opes = actions[0, :]
        mas = actions[1, :]
        #find wait index and active index
        active_indices = (actions[0, :] != -1).nonzero(as_tuple=True)[0]
        active_batch_idxes = batch_idxes[active_indices] #FIXME active_indices
        opes = opes[active_indices]
        mas = mas[active_indices]

        # Get the global indices for operations and machines in the action
        #active batch
        active_opes_batch=self.get_opes_indices(active_batch_idxes,opes)
        active_mas_batch=self.get_mas_indices(active_batch_idxes,mas)
        #all global machine indices of the batches being updated
        mas_batch_indices = self.data['mas'].batch
        active_mask = torch.isin(mas_batch_indices, active_batch_idxes)
        mask = torch.isin(mas_batch_indices, batch_idxes)
        selected_mas=mask.nonzero(as_tuple=False).squeeze()
        #all global operation indices of the batches being updated
        opes_batch_indices = self.data['opes'].batch
        active_mask = torch.isin(opes_batch_indices, active_batch_idxes)
        active_selected_opes = active_mask.nonzero(as_tuple=False).squeeze()
        mask = torch.isin(opes_batch_indices, batch_idxes)
        selected_opes=mask.nonzero(as_tuple=False).squeeze()       


        #update opes
        ## Update operation features: status, number of required operations
        self.data['opes'].x[active_opes_batch,1]=1
        ope_req_1d=ope_req_num_batch[active_batch_idxes].flatten()
        ope_req_1d=ope_req_1d[ope_req_1d!=100]
        self.data['opes'].x[active_selected_opes, 0] = ope_req_1d.float()
        ## Update waiting times and remaining processing times for operations
        ## convert ready_time,end_time to corresponding shape
        time_batched =torch.repeat_interleave(time[batch_idxes], self.ope_num[batch_idxes], dim=0)
        max_cols = ready_time.size(1)
        cumulative_indices = torch.arange(max_cols).expand(len(batch_idxes), max_cols)
        length_mask = cumulative_indices < self.ope_num[batch_idxes, None]
        ready_time = ready_time[batch_idxes][length_mask]
        end_time =end_time[batch_idxes][length_mask]
        
        self.data['opes'].x[selected_opes,3]=(time_batched-ready_time)*(ready_time>0)#update waiting time
        self.data['opes'].x[selected_opes,4]=(end_time-time_batched)*(self.data['opes'].x[selected_opes,1])#update remaining processing time

        #update machines:available time, utilization, status,waiting time, remaining processing time
        self.data['mas'].x[active_mas_batch,1]=machine[active_batch_idxes,mas,1]#available time
        #caculate utiliz
        utiliz=machine[batch_idxes,:,2]
        cur_time = time[batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(time[batch_idxes, None] + 1e-9)
        utiliz= utiliz.view(-1)
        time_flat=torch.repeat_interleave(time[batch_idxes], self.mas_num[batch_idxes], dim=0)
        self.data['mas'].x[selected_mas,2]=utiliz

        #status,waiting time, remaining processing time
        self.data['mas'].x[selected_mas, 3] = (self.data['mas'].x[selected_mas, 1] >time_flat).float()#if operating
        self.data['mas'].x[selected_mas, 4]=abs((time_flat- self.data['mas'].x[selected_mas, 1] )*(1-self.data['mas'].x[selected_mas, 3]))#waiting
        self.data['mas'].x[selected_mas, 5]=abs(-(time_flat - self.data['mas'].x[selected_mas, 1] )*self.data['mas'].x[selected_mas, 3])#Remaining processing time
    
        #update combinations: estimated end time
        self.data['combs'].x[:, 0] = flatten_padded_tensor(self.data['combs'].ptr, combs_time_batch).flatten()
        edge_min_combs_in_job =  self.data['combs'].x[ self.data['combs', 'belong_job', 'jobs'].edge_index[0], 0].squeeze(-1)
        
        if len(edge_min_combs_in_job.shape) != 0:
            min_combs_in_job = scatter(edge_min_combs_in_job,  self.data['combs', 'belong_job', 'jobs'].edge_index[1], dim=0, reduce="min")[self.data['combs', 'belong_job', 'jobs'].edge_index[1]]
            min_combs_in_job_to_combs = scatter(min_combs_in_job, self.data['combs', 'belong_job', 'jobs'].edge_index[0], dim=0, reduce="min")
            self.data['combs'].x[:min_combs_in_job_to_combs.size(-1), 1] = min_combs_in_job_to_combs / (self.data['combs'].x[:min_combs_in_job_to_combs.size(-1), 0] * 1.25 - min_combs_in_job_to_combs + 1e-4)
            self.data['jobs'].x[:, 0] = flatten_padded_tensor(self.data['jobs'].ptr, torch.exp((job_estimate_time - time.unsqueeze(1)) / (job_estimate_time.max() * 1.25 - job_estimate_time + 1e-4)))
        #update edge subgraph
        remain_combs = torch.where(combs_id_batch.sum(dim = -2) > 0, 1, 0)
        self.update_edge_sub_graph(remain_opes, remain_combs)


    def get_data(self):
        '''return batched graph'''
        return self.data

