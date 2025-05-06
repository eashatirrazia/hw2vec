import os, sys
from hw2vec.hw2vec.config import Config
from hw2vec.hw2vec.hw2graph import *
from hw2vec.hw2vec.graph2vec.models import *
import random
import matplotlib.pyplot as plt
from pathlib import Path


cfg = Config(sys.argv[1:])


''' prepare graph data '''
if not cfg.data_pkl_path.exists():
    ''' converting graph using hw2graph '''
    nx_graphs = []
    hw2graph = HW2GRAPH(cfg)
    print(f"Finding hardware project folders in {cfg.raw_dataset_path}")
    for hw_project_path in hw2graph.find_hw_project_folders():
        hw_graph = hw2graph.code2graph(hw_project_path)
        nx_graphs.append(hw_graph)
    
    data_proc = DataProcessor(cfg)
    for hw_graph in nx_graphs:
        data_proc.process(hw_graph)
    data_proc.cache_graph_data(cfg.data_pkl_path)
    
else:
    ''' reading graph data from cache '''
    data_proc = DataProcessor(cfg)
    print(f"Reading graph data from {cfg.data_pkl_path}")
    data_proc.read_graph_data_from_cache(cfg.data_pkl_path)
    

''' prepare dataset '''
TROJAN = 1
NON_TROJAN = 0

all_graphs = data_proc.get_graphs()
print(f"Total number of graphs: {len(all_graphs)}")

for data in all_graphs:
    if "TjFree" == data.hw_type:
        data.label = NON_TROJAN
    else:
        data.label = TROJAN

train_graphs, test_graphs = data_proc.split_dataset(ratio=cfg.ratio, seed=cfg.seed, dataset=all_graphs)

train_trojan_count = sum(1 for graph in train_graphs if graph.label == 1)
train_non_trojan_count = len(train_graphs) - train_trojan_count

test_trojan_count = sum(1 for graph in test_graphs if graph.label == 1)
test_non_trojan_count = len(test_graphs) - test_trojan_count

print(f"Train dataset: {train_trojan_count} Trojan, {train_non_trojan_count} Non-Trojan")
print(f"Test dataset: {test_trojan_count} Trojan, {test_non_trojan_count} Non-Trojan")

train_trojans = [graph.name for graph in train_graphs if graph.label == 1]
test_trojans = [graph.name for graph in test_graphs if graph.label == 1]

print(f"Train Trojans: {train_trojans}")
print(f"Test Trojans: {test_trojans}")

train_loader = DataLoader(train_graphs, shuffle=True, batch_size=cfg.batch_size)
valid_loader = DataLoader(test_graphs, shuffle=True, batch_size=1)



# ''' model configuration '''
# print("Model Path: ", cfg.model_path)
# model = GRAPH2VEC(cfg)
# if cfg.model_path != "":
#     model_path = Path(cfg.model_path)
#     if model_path.exists():
#         model.load_model(str(model_path/"model.cfg"), str(model_path/"model.pth"))
#         print("Model loaded from: ", model_path)
# else:
#     convs = [
#         GRAPH_CONV("gcn", data_proc.num_node_labels, cfg.hidden),
#     ]
#     for i in range(cfg.num_layer):
#         convs.append(GRAPH_CONV("gcn", cfg.hidden, cfg.hidden))
        
#     model.set_graph_conv(convs)

#     pool = GRAPH_POOL("sagpool", cfg.hidden, cfg.poolratio)
#     model.set_graph_pool(pool)

#     readout = GRAPH_READOUT("max")
#     model.set_graph_readout(readout)

#     output = nn.Linear(cfg.hidden, cfg.embed_dim)
#     model.set_output_layer(output)


model_type = cfg.model_type
''' model configuration '''
print("Model Path: ", cfg.model_path)
model = GRAPH2VEC(cfg)
if cfg.model_path != "":
    model_path = Path(cfg.model_path)
    if model_path.exists():
        model.load_model(str(model_path/"model.cfg"), str(model_path/"model_trained.pth"))
        print("Model loaded from: ", model_path)
else:
    convs = [
        GRAPH_CONV(model_type, data_proc.num_node_labels, cfg.hidden),
        GRAPH_CONV(model_type, cfg.hidden, cfg.hidden)
    ]
    model.set_graph_conv(convs)

    pool = GRAPH_POOL("sagpool", cfg.hidden, cfg.poolratio)
    model.set_graph_pool(pool)

    readout = GRAPH_READOUT("max")
    model.set_graph_readout(readout)

    output = nn.Linear(cfg.hidden, cfg.embed_dim)
    model.set_output_layer(output)

''' training '''


model_name = cfg.model_name

base_save_path = cfg.model_save_path  # Convert the base path to a Path object

config_file_path = os.path.join(base_save_path, f"{model_name}.json")
model_file_path = os.path.join(base_save_path, f"{model_name}.pth")

model.to(cfg.device)
trainer = GraphTrainer(cfg, class_weights=data_proc.get_class_weights(train_graphs))
trainer.build(model)
loss_history = trainer.train(train_loader, valid_loader, model_cfg_path=config_file_path, model_path=model_file_path)


plt.figure()
plt.plot(loss_history)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title(f"loss vs. epochs with lr={cfg.learning_rate}")
plt.savefig(f"trained_models/{model_name}.png")
plt.show()



# Ensure the base save directory exists
base_save_path= Path(base_save_path)
base_save_path.mkdir(parents=True, exist_ok=True)

# Now use these Path objects for saving
trainer.save_model(str(config_file_path), str(model_file_path))

print(f"Configuration saved to: {config_file_path}")
print(f"Model saved to: {model_file_path}")

''' evaluating and inspecting '''
trainer.evaluate(cfg.epochs, train_loader, valid_loader)
vis_loader = DataLoader(all_graphs, shuffle=False, batch_size=1)
trainer.visualize_embeddings(vis_loader, "./")