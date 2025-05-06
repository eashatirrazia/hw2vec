#!/usr/bin/env python
#title           :trainers.py
#description     :This file includes the trainers of hw2vec.
#author          :Shih-Yuan Yu
#date            :2021/03/05
#version         :0.2
#notes           :
#python_version  :3.6
#==============================================================================
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pkl
import re

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm

from torch_geometric.data import Data, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

# from hw2vec.graph2vec.models import *
# from hw2vec.utilities import *

from hw2vec.hw2vec.graph2vec.models import *
from hw2vec.hw2vec.utilities import *

class BaseTrainer:
    def __init__(self, cfg):
        self.config = cfg
        self.min_test_loss = np.Inf
        self.task = None
        self.metrics = {}
        self.model = None
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def build(self, model, path=None):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=8e-5)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=300)
    def visualize_embeddings(self, data_loader, path=None):
        save_path = "./visualize_embeddings/" if path is None else Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        embeddings, hw_names = self.get_embeddings(data_loader)

        with open(str(save_path / "vectors.tsv"), "w") as vectors_file, \
             open(str(save_path / "metadata.tsv"), "w") as metadata_file:

            for embed, name in zip(embeddings, hw_names):
                vectors_file.write("\t".join([str(x) for x in embed.detach().cpu().numpy()[0]]) + "\n")
                metadata_file.write(name+"\n")

    def get_embeddings(self, data_loader):
        embeds = []
        hw_names = []

        with torch.no_grad():
            self.model.eval()

            for data in data_loader:
                data.to(self.config.device)
                embed_x, _ = self.model.embed_graph(data.x, data.edge_index, data.batch)
                embeds.append(embed_x)
                hw_names += data.hw_namcfge

        return embeds, hw_names

    def metric_calc(self, loss, labels, preds, header):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        conf_mtx = str(confusion_matrix(labels, preds)).replace('\n', ',')
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")

        self.metric_print(loss, acc, f1, conf_mtx, precision, recall, header)

        if header == "test " and (self.min_test_loss >= loss):
            self.min_test_loss = loss
            self.metrics["acc"] = acc
            self.metrics["f1"] = f1
            self.metrics["conf_mtx"] = conf_mtx
            self.metrics["precision"] = precision
            self.metrics["recall"] = recall

    def metric_print(self, loss, acc, f1, conf_mtx, precision, recall, header):
        print("%s loss: %4f" % (header, loss) +
            ", %s accuracy: %.4f" % (header, acc) +
            ", %s f1 score: %.4f" % (header, f1) +
            ", %s confusion_matrix: %s" % (header, conf_mtx) +
            ", %s precision: %.4f" % (header, precision) +
            ", %s recall: %.4f" % (header, recall))

class PairwiseGraphTrainer(BaseTrainer):
    ''' trainer for graph classification ''' 
    def __init__(self, cfg):
        super().__init__(cfg)
        self.task = "IP"
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-08).to(self.config.device)
        self.cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.5).to(self.config.device)
    
    def train(self, train_loader, test_loader):
        tqdm_bar = tqdm(range(self.config.epochs))

        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0
            
            for data in train_loader:
                self.optimizer.zero_grad()
                graph1, graph2, labels = data[0].to(self.config.device), data[1].to(self.config.device), data[2].to(self.config.device)

                loss_train = self.train_epoch_ip(graph1, graph2, labels)
                loss_train.backward()
                self.optimizer.step()

                acc_loss_train += loss_train.detach().cpu().numpy()

            self.lr_scheduler.step()
            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.1f}, lr: {:.4e}'.format(epoch_idx, acc_loss_train/len(train_loader), self.lr_scheduler.get_lr()))

            if epoch_idx % self.config.test_step == 0:
                self.evaluate(epoch_idx, train_loader, test_loader)
    
    # @profileit
    def train_epoch_ip(self, graph1, graph2, labels):
        g_emb_1, _ = self.model.embed_graph(graph1.x, graph1.edge_index, batch=graph1.batch)
        g_emb_2, _ = self.model.embed_graph(graph2.x, graph2.edge_index, batch=graph2.batch)
        
        g_emb_1 = self.model.mlp(g_emb_1)
        g_emb_2 = self.model.mlp(g_emb_2)

        loss_train = self.cos_loss(g_emb_1, g_emb_2, labels)
        return loss_train

    # @profileit
    def inference_epoch_ip(self, graph1, graph2):
        g_emb_1, _ = self.model.embed_graph(graph1.x, graph1.edge_index, batch=graph1.batch)
        g_emb_2, _ = self.model.embed_graph(graph2.x, graph2.edge_index, batch=graph2.batch)

        g_emb_1 = self.model.mlp(g_emb_1)
        g_emb_2 = self.model.mlp(g_emb_2)

        similarity = self.cos_sim(g_emb_1, g_emb_2)
        return g_emb_1, g_emb_2, similarity

    def inference(self, data_loader):
        labels = []
        outputs = []
        total_loss = 0
        with torch.no_grad():
            self.model.eval()
            
            for data in data_loader:
                graph1, graph2, labels_batch = data[0].to(self.config.device), data[1].to(self.config.device), data[2].to(self.config.device)
                    
                g_emb_1, g_emb_2, similarity = self.inference_epoch_ip(graph1, graph2)
                loss = self.cos_loss(g_emb_1, g_emb_2, labels_batch)

                total_loss += loss.detach().cpu().numpy()
                outputs.append(similarity.detach().cpu())
                labels += np.split(labels_batch.detach().cpu().numpy(), len(labels_batch.detach().cpu().numpy()))

            outputs = torch.cat(outputs).detach()
            avg_loss = total_loss / (len(data_loader))

            labels_tensor = (torch.LongTensor(labels)> 0).detach() 
            outputs_tensor = torch.FloatTensor(outputs).detach()
            preds = (outputs > 0.5).detach()

        return avg_loss, labels_tensor, outputs_tensor, preds
    
    def evaluate(self, epoch_idx, train_loader, test_loader):
        train_loss, train_labels, _, train_preds = self.inference(train_loader)
        test_loss, test_labels, _, test_preds = self.inference(test_loader)

        print("")
        print("Mini Test for Epochs %d:"%epoch_idx)

        self.metric_calc(train_loss, train_labels, train_preds, header="train")
        self.metric_calc(test_loss,  test_labels,  test_preds,  header="test ")

        if self.min_test_loss >= test_loss:
            self.model.save_model(str(self.config.model_path_obj/"model.cfg"), str(self.config.model_path_obj/"model.pth"))

        # on final evaluate call
        if(epoch_idx==self.config.epochs):
            self.metric_print(self.min_test_loss, **self.metrics, header="best ")

class GraphTrainer(BaseTrainer):
    ''' trainer for graph classification ''' 
    def __init__(self, cfg, class_weights=None):
        super().__init__(cfg)
        self.task = "TJ"
        if class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:    
            self.loss_func = nn.CrossEntropyLoss(weight=class_weights.float().to(cfg.device))

    def train(self, data_loader, valid_data_loader, model_cfg_path, model_path):
        tqdm_bar = tqdm(range(self.config.epochs))
        total_train_loss_history = []
        best_train_loss = float('inf')

        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0

            for data in data_loader:
                self.optimizer.zero_grad()
                data.to(self.config.device)

                loss_train = self.train_epoch_tj(data)
                loss_train.backward()
                self.optimizer.step()
                acc_loss_train += loss_train.detach().cpu().numpy()

            avg_loss_train = acc_loss_train / len(data_loader)
            total_train_loss_history.append(avg_loss_train)

            self.lr_scheduler.step()
            tqdm_bar.set_description(
                'Epoch: {:04d}, loss_train: {:.4f}, lr: {:.4e}'.format(
                    epoch_idx, avg_loss_train, self.lr_scheduler.get_last_lr()[0]
                )
            )

            # Save model if it's the best so far
            if avg_loss_train < best_train_loss:
                best_train_loss = avg_loss_train
                best_cfg_path = model_cfg_path.replace('.json', '_best.json')
                best_model_path = model_path.replace('.pth', '_best.pth')
                self.save_model(best_cfg_path, best_model_path)
                print(f"loss changed from {avg_loss_train} to {best_train_loss}")
                print("Saving the model ....")

            if epoch_idx % self.config.test_step == 0:
                self.evaluate(epoch_idx, data_loader, valid_data_loader,model_cfg_path, model_path)

        return total_train_loss_history
                    
    def save_model(self, model_cfg_path, model_path):
        # Save the model's weights
        torch.save(self.model.state_dict(), model_path)

        # Parse the model description
        model_description = self.parse_model_description(str(self.model))

        # Save the model description to a JSON config file
        with open(model_cfg_path, 'w') as f:
            json.dump(model_description, f, indent=4)



    # def parse_model_description(self, model_str):
    #     convs = []
    #     pool = None
    #     readout = "max"  # default assumption
    #     fc = []

    #     lines = model_str.splitlines()
    #     i = 0
    #     while i < len(lines):
    #         line = lines[i].strip()

    #         # Match repeated convs like (1-4): 4 x GRAPH_CONV...
    #         repeat_match = re.search(r'\((\d+)-(\d+)\):\s+(\d+)\s+x\s+GRAPH_CONV', line)
    #         if repeat_match:
    #             start_idx, end_idx, count = map(int, repeat_match.groups())
    #             i += 1
    #             # Skip to the line that contains the GCNConv description
    #             while i < len(lines) and "GCNConv" not in lines[i]:
    #                 i += 1
    #             if i < len(lines):
    #                 conv_line = lines[i]
    #                 conv_match = re.search(r'GCNConv\((\d+),\s*(\d+)\)', conv_line)
    #                 if conv_match:
    #                     in_dim, out_dim = map(int, conv_match.groups())
    #                     convs.extend([["gcn", in_dim, out_dim]] * count)

    #         # Match individual GCNConv layers like (0): ...
    #         elif "GCNConv" in line:
    #             conv_match = re.search(r'GCNConv\((\d+),\s*(\d+)\)', line)
    #             if conv_match:
    #                 in_dim, out_dim = map(int, conv_match.groups())
    #                 convs.append(["gcn", in_dim, out_dim])

    #         # Match SAGPooling
    #         elif 'SAGPooling' in line:
    #             sag_match = re.search(r'SAGPooling\(\w+,\s*(\d+),\s*ratio=([\d.]+)', line)
    #             if sag_match:
    #                 dim, ratio = int(sag_match.group(1)), float(sag_match.group(2))
    #                 pool = ["sagpool", dim, ratio]

    #         # Match Fully Connected layer
    #         elif 'Linear' in line:
    #             fc_match = re.search(r'in_features=(\d+),\s*out_features=(\d+)', line)
    #             if fc_match:
    #                 fc = [int(fc_match.group(1)), int(fc_match.group(2))]

    #         i += 1

    #     return {
    #         "convs": convs,
    #         "pool": pool,
    #         "readout": readout,
    #         "fc": fc
    #     }


    def parse_model_description(self, model_str):
        convs = []
        pool = None
        readout = "max"  # default assumption
        fc = []

        lines = model_str.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Match repeated convs like (1-4): 4 x GRAPH_CONV...
            repeat_match = re.search(r'\((\d+)-(\d+)\):\s+(\d+)\s+x\s+GRAPH_CONV', line)
            if repeat_match:
                start_idx, end_idx, count = map(int, repeat_match.groups())
                i += 1
                # Skip to the line that contains the GCNConv description
                while i < len(lines) and "GCNConv" not in lines[i]:
                    i += 1
                if i < len(lines):
                    conv_line = lines[i]
                    conv_match = re.search(r'GCNConv\((\d+),\s*(\d+)\)', conv_line)
                    if conv_match:
                        in_dim, out_dim = map(int, conv_match.groups())
                        convs.extend([["gcn", in_dim, out_dim]] * count)

            # Match individual GCNConv layers like (0): ...
            elif "GCNConv" in line:
                conv_match = re.search(r'GCNConv\((\d+),\s*(\d+)\)', line)
                if conv_match:
                    in_dim, out_dim = map(int, conv_match.groups())
                    convs.append(["gcn", in_dim, out_dim])
            elif "GINConv" in line:
                conv_match = re.search(r'GINConv\(nn=Linear\(in_features=(\d+),\s*out_features=(\d+)', line)
                if conv_match:
                    in_dim, out_dim = map(int, conv_match.groups())
                    convs.append(["gin", in_dim, out_dim])
            # Match AGNNConv layers
            elif "AGNNConv" in line:
                # Look ahead a few lines for the Linear layer info
                j = i - 1  # Start from the previous line to check for the Linear component
                linear_found = False
                while j >= 0 and j >= i - 3:  # Check up to 3 lines before
                    if "Linear" in lines[j]:
                        agnn_match = re.search(r'Linear\(in_features=(\d+),\s*out_features=(\d+)', lines[j])
                        if agnn_match:
                            in_dim, out_dim = map(int, agnn_match.groups())
                            convs.append(["agnn", in_dim, out_dim])
                            linear_found = True
                            break
                    j -= 1
                
                # Also check a few lines ahead if not found before
                if not linear_found:
                    j = i + 1
                    while j < len(lines) and j <= i + 3:  # Check up to 3 lines after
                        if "Linear" in lines[j]:
                            agnn_match = re.search(r'Linear\(in_features=(\d+),\s*out_features=(\d+)', lines[j])
                            if agnn_match:
                                in_dim, out_dim = map(int, agnn_match.groups())
                                convs.append(["agnn", in_dim, out_dim])
                                break
                        j += 1

            # Match SAGPooling
            elif 'SAGPooling' in line:
                sag_match = re.search(r'SAGPooling\(\w+,\s*(\d+),\s*ratio=([\d.]+)', line)
                if sag_match:
                    dim, ratio = int(sag_match.group(1)), float(sag_match.group(2))
                    pool = ["sagpool", dim, ratio]

            # Match Fully Connected layer
            elif 'Linear' in line:
                fc_match = re.search(r'in_features=(\d+),\s*out_features=(\d+)', line)
                if fc_match:
                    fc = [int(fc_match.group(1)), int(fc_match.group(2))]

            i += 1

        return {
            "convs": convs,
            "pool": pool,
            "readout": readout,
            "fc":fc
            }


    # @profileit
    def train_epoch_tj(self, data):
        output, _ = self.model.embed_graph(data.x, data.edge_index, data.batch)
        output = self.model.mlp(output)
        output = F.log_softmax(output, dim=1)

        loss_train = self.loss_func(output, data.label)
        return loss_train

    # @profileit
    def inference_epoch_tj(self, data):
        output, attn = self.model.embed_graph(data.x, data.edge_index, data.batch)
        output = self.model.mlp(output)
        output = F.log_softmax(output, dim=1)

        loss = self.loss_func(output, data.label)
        return loss, output, attn
                
    def inference(self, data_loader):
        labels = []
        outputs = []
        node_attns = []
        total_loss = 0
        folder_names = []
        
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(data_loader):
                data.to(self.config.device)

                loss, output, attn = self.inference_epoch_tj(data)
                total_loss += loss.detach().cpu().numpy()

                outputs.append(output.cpu())
                
                if 'pool_score' in attn:
                    node_attn = {}
                    node_attn["original_batch"] = data.batch.detach().cpu().numpy().tolist()
                    node_attn["pool_perm"] = attn['pool_perm'].detach().cpu().numpy().tolist()
                    node_attn["pool_batch"] = attn['batch'].detach().cpu().numpy().tolist()
                    node_attn["pool_score"] = attn['pool_score'].detach().cpu().numpy().tolist()
                    node_attns.append(node_attn)

                labels += np.split(data.label.cpu().numpy(), len(data.label.cpu().numpy()))

            outputs = torch.cat(outputs).reshape(-1,2).detach()
            avg_loss = total_loss / (len(data_loader))

            labels_tensor = torch.LongTensor(labels).detach()
            # print("Labels:", labels_tensor)
            outputs_tensor = torch.FloatTensor(outputs).detach()
            preds = outputs_tensor.max(1)[1].type_as(labels_tensor).detach()
            # print("Preds:", preds)

        return avg_loss, labels_tensor, outputs_tensor, preds, node_attns

    def evaluate(self, epoch_idx, data_loader, valid_data_loader,model_cfg_path=None, model_path=None):
        train_loss, train_labels, _, train_preds, train_node_attns = self.inference(data_loader)
        test_loss, test_labels, _, test_preds, test_node_attns = self.inference(valid_data_loader)

        print("")
        print("Mini Test for Epochs %d:"%epoch_idx)

        self.metric_calc(train_loss, train_labels, train_preds, header="train")
        self.metric_calc(test_loss,  test_labels,  test_preds,  header="test ")

        if self.min_test_loss >= test_loss:
            # self.model.save_model(str(self.config.model_path_obj/"model.cfg"), str(self.config.model_path_obj/"model.pth"))
            if model_cfg_path and model_path:
                best_cfg_path = model_cfg_path.replace('.json', '_best_test.json')
                best_model_path = model_path.replace('.pth', '_best_test.pth')
                self.save_model(best_cfg_path, best_model_path)
                print("Saving the best test model ....")
            print(f"loss changed from {self.min_test_loss} to {test_loss}")

            #TODO: store the attn_weights right here. 

        # on final evaluate call
        if(epoch_idx==self.config.epochs):
            self.metric_print(self.min_test_loss, **self.metrics, header="best ")


class Evaluator(BaseTrainer):
    def __init__(self, cfg, task):
        super().__init__(cfg)
        self.task = task