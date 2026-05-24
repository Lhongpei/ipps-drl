import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import  GATv2Conv, JumpingKnowledge, HeteroConv
from torch.nn import ModuleList
from torch_geometric.data import HeteroData
from omegaconf import OmegaConf
# from torch_geometric.nn import HANConv, HGTConv
from torch_geometric.nn import GraphNorm

class GraphEmbedding(torch.nn.Module):
    """Graph Embedding using GATv2Conv for heterogeneous graph data.
    
    The structure of the graph is:
    - Node types: 
        - opes
        - mas
    - Edge types: 
        - pre: opes -> opes, if opes has a pre-sub relationship with opes
        - proc: mas -> opes, if mas can process opes
        - forall: mas -> mas, link all mas nodes
    
    Args:
        config (object): Configuration object containing the graph embedding parameters.
    
    Attributes:
        num_heads (list): List of integers representing the number of attention heads for each layer.
        num_layers (int): Number of layers in the graph embedding model.
        hidden_dim (int): Hidden dimension size for the graph embedding model.
        dropout (float): Dropout rate for the graph embedding model.
        jk (str): Jumping Knowledge mode for combining node representations across layers.
        convs (torch.nn.ModuleList): List of HeteroConv layers for graph convolution.
        jk (torch.nn.Module): JumpingKnowledge module for combining node representations across layers.
    
    Forward:
    - Input: HeteroData
    - Output: HeteroData
    """
    def __init__(self, config):
        super(GraphEmbedding, self).__init__()
        self.num_heads = config.graph_embedding.num_heads
        self.num_layers = len(self.num_heads)
        self.hidden_dim = config.graph_embedding.hidden_dim
        self.dropout = config.graph_embedding.dropout
        jk = config.graph_embedding.jk
        ope_dim = config.graph_embedding.opes_dim
        mas_dim = config.graph_embedding.mas_dim
        comb_dim = config.graph_embedding.combs_dim
        job_dim = config.graph_embedding.jobs_dim
        proc_dim = config.graph_embedding.proc_dim
        self.graph_norm = GraphNorm(self.hidden_dim) 
        assert jk in ['cat', 'max', 'lstm', None]
        for i in self.num_heads:
            assert self.hidden_dim % i == 0, "Hidden dim must be divisible by the number of heads."
            
        self.convs = ModuleList()
        self.convs.append(HeteroConv({
            ('opes', 'pre', 'opes'): GATv2Conv(ope_dim, self.hidden_dim//self.num_heads[0], add_self_loops=False, heads=self.num_heads[0], dropout=self.dropout),
            ('mas', 'proc', 'opes'): GATv2Conv((mas_dim, ope_dim), self.hidden_dim//self.num_heads[0], add_self_loops=False, heads=self.num_heads[0], edge_dim=proc_dim, dropout=self.dropout),
            ('opes', 'proc_rev', 'mas'): GATv2Conv((ope_dim, mas_dim), self.hidden_dim//self.num_heads[0], add_self_loops=False, heads=self.num_heads[0], edge_dim=proc_dim, dropout=self.dropout),
            ('opes', 'belong_comb', 'combs'): GATv2Conv((ope_dim, comb_dim), self.hidden_dim//self.num_heads[0], add_self_loops=False, heads=self.num_heads[0], dropout=self.dropout),
            ('combs', 'belong_comb_rev', 'opes'): GATv2Conv((comb_dim, ope_dim), self.hidden_dim//self.num_heads[0], add_self_loops=False, heads=self.num_heads[0], dropout=self.dropout),
            ('combs', 'belong_job', 'jobs'): GATv2Conv((comb_dim, job_dim), self.hidden_dim//self.num_heads[0], add_self_loops=False, heads=self.num_heads[0], dropout=self.dropout),
            ('jobs', 'belong_job_rev', 'combs'): GATv2Conv((job_dim, comb_dim), self.hidden_dim//self.num_heads[0], add_self_loops=False, heads=self.num_heads[0], dropout=self.dropout)
        }))
        
        for i in range(1, self.num_layers):
            self.convs.append(HeteroConv({
                ('opes', 'pre', 'opes'): GATv2Conv(self.hidden_dim, self.hidden_dim//self.num_heads[i], add_self_loops=False, heads=self.num_heads[i], dropout=self.dropout),
                ('mas', 'proc', 'opes'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads[i], add_self_loops=False, heads=self.num_heads[i], edge_dim=proc_dim, dropout=self.dropout),
                ('opes', 'proc_rev', 'mas'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads[i], add_self_loops=False, heads=self.num_heads[i], edge_dim=proc_dim, dropout=self.dropout),
                ('opes', 'belong_comb', 'combs'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads[i], add_self_loops=False, heads=self.num_heads[i], dropout=self.dropout),
                ('combs', 'belong_comb_rev', 'opes'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads[i], add_self_loops=False, heads=self.num_heads[i], dropout=self.dropout),
                ('combs', 'belong_job', 'jobs'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads[i], add_self_loops=False, heads=self.num_heads[i], dropout=self.dropout),
                ('jobs', 'belong_job_rev', 'combs'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads[i], add_self_loops=False, heads=self.num_heads[i], dropout=self.dropout)
            }))
            
        self.jk = JumpingKnowledge(mode=jk, channels=self.hidden_dim, num_layers=self.num_layers) if jk in ['cat', 'max', 'lstm'] else None
        
    def forward(self, hetero_data: HeteroData) -> HeteroData:
        """
        Forward pass of the graph embedding model.
        
        Args:
            hetero_data (HeteroData): Input graph data.
            
        Returns:
            hetero_data (HeteroData): Output graph data.
        """
        ope_output = []
        mas_output = []
        combs_output = []
        jobs_output = []
        x_dict = hetero_data.x_dict
        batch_dict = hetero_data.batch_dict
        edge_index_dict = hetero_data.edge_index_dict
        edge_attr_dict = hetero_data.edge_attr_dict
        i = 0
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.graph_norm(x, batch_dict[key]) for key, x in x_dict.items()}

            ope_output.append(x_dict['opes'])
            mas_output.append(x_dict['mas'])
            combs_output.append(x_dict['combs'])
            jobs_output.append(x_dict['jobs'])
            i +=1
        if self.jk is not None:
            x_opes = self.jk(ope_output)
            x_mas = self.jk(mas_output)
            x_combs = self.jk(combs_output)
            x_jobs = self.jk(jobs_output)
        else:
            x_opes = ope_output[-1]
            x_mas = mas_output[-1]
            x_combs = combs_output[-1]
            x_jobs = jobs_output[-1]

        hetero_data['opes'].x = x_opes
        hetero_data['mas'].x = x_mas
        hetero_data['combs'].x = x_combs
        hetero_data['jobs'].x = x_jobs
        return hetero_data
    


class Actor(nn.Module):
    """Actor model using a simple MLP.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - 'hidden_dims' (list of int): List of hidden layer dimensions.
            - 'in_dim' (int): Dimension of input features.
    """
    def __init__(self, config: dict):
        super(Actor, self).__init__()

        in_dim = 6 * config.graph_embedding.hidden_dim
        if config.graph_embedding.jk == 'cat':
            in_dim *= len(config.graph_embedding.num_heads)
        hidden_dims = config.actor.hidden_dims
        out_dim = 1
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            #layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        
        self.actor = nn.Sequential(*layers)
        self.init_weights()
        
    def init_weights(self):
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor):
        """Forward pass of the actor model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.actor(x)

    
class Critic(nn.Module):
    """Actor model using a simple MLP.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - 'hidden_dims' (list of int): List of hidden layer dimensions.
            - 'in_dim' (int): Dimension of input features.
    """
    def __init__(self, config: dict):
        super(Critic, self).__init__()
        in_dim = 3 * config.graph_embedding.hidden_dim
        if config.graph_embedding.jk == 'cat':
            in_dim *= len(config.graph_embedding.num_heads)
        hidden_dims = config.critic.hidden_dims
        out_dim = 1
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        
        self.critic = nn.Sequential(*layers)
        self.init_weights()
        
    def init_weights(self):
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, x: torch.Tensor):
        """Forward pass of the critic network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.critic(x)


if __name__ == '__main__':
    config = OmegaConf.load("./config.yaml")
    GraphEmbedding(config.nn_paras.graph_embedding)