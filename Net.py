import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nn_g
import config

class Block(nn.Module):

    def __init__(self,in_channels, hidden_channels, ratio, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.attention = nn_g.GATConv(in_channels, hidden_channels, num_layers= config.num_layers, heads = config.num_heads, dropout= config.dropout)
        self.Linear = nn.Linear(config.num_heads * hidden_channels, hidden_channels)
        self.pool = nn_g.TopKPooling(hidden_channels, ratio = ratio)
    
    def forward(self, x, edge_index, edge, batch_index):
        x = self.attention(x, edge_index)
        x = self.Linear(x)
        x, edge_index, edge , batch_index, _ , _ = self.pool(x, edge_index, edge, batch_index)

        x1 = torch.cat([nn_g.global_mean_pool(x,batch_index), nn_g.global_max_pool(x, batch_index)], dim = -1)

        return x, edge_index, edge, batch_index, x1

class GNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.block1 = Block(config.number_features, config.embeding_size, ratio =  config.top_k_ratio)
        self.block2 = Block(config.embeding_size, config.embeding_size, ratio =  config.top_k_ratio)
        self.block3 = Block(config.embeding_size, config.embeding_size, ratio=  config.top_k_ratio)
        self.block4 = Block(config.embeding_size, config.embeding_size, ratio= config.top_k_ratio)

        self.fc = nn.Sequential(nn.Linear(2*config.embeding_size, config.fc_size), 
                                nn.ReLU(), 
                                nn.Dropout(0.5), 
                                nn.Linear(config.fc_size, config.fc_size // 2),
                                nn.ReLU(), 
                                nn.Dropout(0.5),
                                nn.Linear(config.fc_size//2, 1))

    def forward(self, x, edge_index, edge, batch_index):
        keep_x = []
        x, edge_index, edge, batch_index, x1 = self.block1(x, edge_index, edge,  batch_index)
        keep_x.append(x1.unsqueeze(1))
        
        x, edge_index, edge, batch_index, x1 = self.block2(x, edge_index, edge, batch_index)
        keep_x.append(x1.unsqueeze(1))

        x, edge_index, edge, batch_index, x1 = self.block3(x, edge_index, edge, batch_index)
        keep_x.append(x1.unsqueeze(1))

        x = torch.cat(keep_x, dim = 1).sum(1)

        out = self.fc(x)

        return out.squeeze()
