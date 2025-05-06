#!/usr/bin/env python
#title           :models.py
#description     :This file includes the models of hw2vec.
#author          :Easha Tir Razia
#date            :2021/03/05
#version         :0.2
#notes           :
#python_version  :3.6
#==============================================================================
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv, SAGPooling, TopKPooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torchinfo import summary

class GRAPH_CONV(nn.Module):
    def __init__(self, type, in_channels, out_channels):
        super(GRAPH_CONV, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type == "gcn":
            self.graph_conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.graph_conv(x, edge_index)

class GRAPH_POOL(nn.Module):
    def __init__(self, type, in_channels, poolratio):
        super(GRAPH_POOL, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.poolratio = poolratio
        if self.type == "sagpool":
            self.graph_pool = SAGPooling(in_channels, ratio=poolratio)
        elif self.type == "topkpool":
            self.graph_pool = TopKPooling(in_channels, ratio=poolratio)
    
    def forward(self, x, edge_index, batch):
        return self.graph_pool(x, edge_index, batch=batch)

class GRAPH_READOUT(nn.Module):
    def __init__(self, type):
        super(GRAPH_READOUT, self).__init__()
        self.type = type
    
    def forward(self, x, batch):
        if self.type == "max":
            return global_max_pool(x, batch)
        elif self.type == "mean":
            return global_mean_pool(x, batch)
        elif self.type == "add":
            return global_add_pool(x, batch)


class GRAPH2VEC(nn.Module):
    
    ''' 
        For users who want to develop their own network architecture, 
        you may use this graph2vec class as template and implement your architecture.
    '''

    def __init__(self, config):
        super(GRAPH2VEC, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.pool1 = None
        self.graph_readout = None
        self.fc = None

    def build_model(self, convs, pool, readout, fc_in, fc_out):
        self.set_graph_conv(convs)
        self.set_graph_pool(pool)
        self.set_graph_readout(readout)
        self.set_output_layer(nn.Linear(fc_in, fc_out))
        
    def set_graph_conv(self, convs):
        self.layers = nn.ModuleList()
        for conv in convs:
            self.layers.append(conv)

    def set_graph_pool(self, pool_layer):
        self.pool1 = pool_layer

    def set_graph_readout(self, typeofreadout):
        self.graph_readout = typeofreadout

    def set_output_layer(self, layer):
        self.fc = layer
    
    def embed_graph(self, x, edge_index, batch):
        attn_weights = dict()
        x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=self.config.dropout, training=self.training)
        x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
            self.pool1(x, edge_index, batch=batch)
        x = self.graph_readout(x, batch)

        attn_weights['batch'] = batch
        return x, attn_weights

    def embed_node(self, x, edge_index):
        x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=self.config.dropout, training=self.training)
        return x

    def mlp(self, x):
        return self.fc(x)

if __name__ == "__main__":
    class Config:
        def __init__(self):
            self.device = torch.device('cpu')  # Or 'cuda' if available
            self.num_feature_dim = 10
            self.dropout = 0.5

    config = Config()

    # Example model configuration
    convs = [GRAPH_CONV('gcn', 10, 16), GRAPH_CONV('gcn', 16, 32)]
    pool = GRAPH_POOL('sagpool', 32, 0.5)
    readout = GRAPH_READOUT('mean')
    fc_in = 32
    fc_out = 2

    model = GRAPH2VEC(config)
    model.build_model(convs, pool, readout, fc_in, fc_out)

    # Create dummy input data
    x = torch.randint(0, 10, (10,)).long()  # Node features
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)  # Edge indices
    batch = torch.zeros(10, dtype=torch.long)  # Batch vector

    # Print model summary
    print(summary(model, input_data=(x, edge_index, batch)))