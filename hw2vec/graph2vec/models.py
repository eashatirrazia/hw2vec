#!/usr/bin/env python
#title           :models.py
#description     :This file includes the models of hw2vec.
#author          :Shih-Yuan Yu
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
from torch_geometric.nn import GCNConv, GINConv, SAGPooling, TopKPooling, AGNNConv, TransformerConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# class GRAPH_CONV(nn.Module):
#     def __init__(self, type, in_channels, out_channels):
#         super(GRAPH_CONV, self).__init__()
#         self.type = type
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if type == "gcn":
#             self.graph_conv = GCNConv(in_channels, out_channels)
#         elif type == "gin":
#             raise Exception("Layer not supperted")
#         elif type == "agnn":
#             raise Exception("Layer not supperted")
#         else:
#             raise Exception("Layer not supperted")


#     def forward(self, x, edge_index):
#         return self.graph_conv(x, edge_index)

class GRAPH_CONV(nn.Module):
    def __init__(self, type, in_channels, out_channels):
        super(GRAPH_CONV, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type == "gcn":
            self.graph_conv = GCNConv(in_channels, out_channels)
        elif type == "gin":
            self.graph_conv = GINConv(Linear(in_channels, out_channels), train_eps=True)
        elif type == "agnn":
            # Fix: AGNNConv only takes a single channels parameter
            # It's a propagation layer that doesn't change feature dimensions
            self.lin = Linear(in_channels, out_channels)
            self.graph_conv = AGNNConv(requires_grad=True)

    def forward(self, x, edge_index):
        if self.type == "agnn":
            # Transform features first, then apply AGNN
            x = self.lin(x)
            return self.graph_conv(x, edge_index)
        else:
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

    def save_model(self, model_config_path, model_weight_path):
        Path(model_config_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_weight_path).parent.mkdir(parents=True, exist_ok=True)
        model_configurations = {}

        convs = [] 
        for layer in self.layers:
            convs.append((layer.type, layer.in_channels, layer.out_channels))
        model_configurations['convs'] = convs

        model_configurations['pool'] = (self.pool1.type, self.pool1.in_channels, self.pool1.poolratio)
        model_configurations['readout'] = self.graph_readout.type
        model_configurations['fc'] = (self.fc.in_features, self.fc.out_features)
        with open(model_config_path, 'w') as f:
            json.dump(model_configurations, f)
        torch.save(self.state_dict(), model_weight_path)

    # def load_model(self, model_config_path, model_weight_path):
    #     with open(model_config_path) as f:
    #         model_configuration = json.load(f)
        
    #     convs = [] 
    #     for setting in model_configuration['convs']:
    #         graph_conv_type, in_channels, out_channels = setting
    #         convs.append(GRAPH_CONV(graph_conv_type, int(in_channels), int(out_channels)))
    #     self.set_graph_conv(convs)

    #     pool_type, pool_in_channels, pool_ratio = model_configuration['pool']
    #     self.set_graph_pool(GRAPH_POOL(pool_type, pool_in_channels, pool_ratio))

    #     self.set_graph_readout(GRAPH_READOUT(model_configuration['readout']))
    #     fc_in_channel, fc_out_channel = model_configuration['fc']
    #     self.set_output_layer(nn.Linear(fc_in_channel, fc_out_channel))

    #     self.load_state_dict(torch.load(model_weight_path))


    def load_model(self, model_config_path, model_weight_path):
        with open(model_config_path) as f:
            model_configuration = json.load(f)
        
        convs = [] 
        for setting in model_configuration['convs']:
            graph_conv_type, in_channels, out_channels = setting
            # print(graph_conv_type, in_channels, out_channels)
            convs.append(GRAPH_CONV(graph_conv_type, int(in_channels), int(out_channels)))
        self.set_graph_conv(convs)

        pool_type, pool_in_channels, pool_ratio = model_configuration['pool']
        # print(pool_type, pool_in_channels, pool_ratio)
        self.set_graph_pool(GRAPH_POOL(pool_type, pool_in_channels, pool_ratio))

        self.set_graph_readout(GRAPH_READOUT(model_configuration['readout']))
        fc_in_channel, fc_out_channel = model_configuration['fc']
        # print(fc_in_channel, fc_out_channel)
        self.set_output_layer(nn.Linear(fc_in_channel, fc_out_channel))

        state_dict = torch.load(model_weight_path)
        new_state_dict = {}

        # Create a mapping from the old key names in the pre-trained model to the new key names in the architecture
        for key, value in state_dict.items():
            new_key = key
            
            # Modify keys for graph convolution layers
            if 'graph_conv.weight' in key:
                new_key = key.replace('graph_conv.weight', 'graph_conv.lin.weight')
            elif 'graph_conv.bias' in key:
                new_key = key.replace('graph_conv.bias', 'graph_conv.lin.bias')
            
            # Modify keys for pooling layers (if applicable)
            elif 'pool1.graph_pool.gnn.lin_l.weight' in key:
                new_key = key.replace('lin_l.weight', 'select.weight')
            elif 'pool1.graph_pool.gnn.lin_l.bias' in key:
                new_key = key.replace('lin_l.bias', 'select.bias')
            elif 'pool1.graph_pool.gnn.lin_r.weight' in key:
                new_key = key.replace('lin_r.weight', 'graph_pool.lin.weight')
            
            # Add other specific key mappings as necessary
            
            # Copy modified key-value pair to the new state_dict
            new_state_dict[new_key] = value


        # Handle potential shape mismatch for certain layers
    
        try:
        # Load the modified state_dict into the model
            self.load_state_dict(new_state_dict, strict=False) 
        except:
            for key in new_state_dict.keys():
                if 'graph_conv.lin.weight' in key:
                    new_state_dict[key] = new_state_dict[key].transpose(0, 1)

            self.load_state_dict(new_state_dict, strict=False) 





        
    def set_graph_conv(self, convs):
        self.layers = []
        
        for conv in convs:
            conv.to(self.config.device)
            self.layers.append(conv)
        self.layers = nn.ModuleList(self.layers)

    def set_graph_pool(self, pool_layer):
        self.pool1 = pool_layer.to(self.config.device)
            
    def set_graph_readout(self, typeofreadout):
        self.graph_readout = typeofreadout

    def set_output_layer(self, layer):
        self.fc = layer.to(self.config.device)
    
    def embed_graph(self, x, edge_index, batch):
        # print(x.shape, edge_index.shape)
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
    
    def predict(self, data):
        self.eval()
        with torch.no_grad():
            x, attn = self.embed_graph(data.x.to(self.config.device), data.edge_index.to(self.config.device), data.batch.to(self.config.device))
            output = self.mlp(x)
            output_tensor = F.log_softmax(output, dim=1)
            preds = output_tensor.max(1)[1]
        return output_tensor, preds