import numpy as np
import random
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim.lr_scheduler as lr_scheduler

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor

from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv

import scipy.sparse
from sklearn.cluster import Birch

import networkx as nx

import argparse
import utils
import os


def parse_args():
    args = argparse.ArgumentParser(description='DGCluster arguments.')
    args.add_argument('--dataset', type=str, default='cora')
    args.add_argument('--lam', type=float, default=0.2)
    args.add_argument('--alp', type=float, default=0.0)
    args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    args.add_argument('--epochs', type=int, default=300)
    args.add_argument('--base_model', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
    args.add_argument('--seed', type=int, default=0)
    args = args.parse_args()
    return args


def load_dataset(dataset_name):
    if dataset_name == 'cora':
        dataset = Planetoid(root='data', name="Cora")
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='data', name="Citeseer")
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='data', name="PubMed")
    elif dataset_name == 'computers':
        dataset = Amazon(root='data', name='Computers')
    elif dataset_name == 'photo':
        dataset = Amazon(root='data', name='Photo')
    elif dataset_name == 'coauthorcs':
        dataset = Coauthor(root='data/Coauthor', name='CS')
    elif dataset_name == 'coauthorphysics':
        dataset = Coauthor(root='data/Coauthor', name='Physics')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} not implemented.')
    return dataset


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, base_model):
        super(GNN, self).__init__()

        if base_model == 'gcn':
            self.conv1 = GCNConv(in_dim, 256)
            self.conv2 = GCNConv(256, 128)
            self.conv3 = GCNConv(128, out_dim)
        elif base_model == 'gat':
            self.conv1 = GATConv(in_dim, 256)
            self.conv2 = GATConv(256, 128)
            self.conv3 = GATConv(128, out_dim)
        elif base_model == 'gin':
            self.conv1 = GINConv(nn.Linear(in_dim, 256))
            self.conv2 = GINConv(nn.Linear(256, 128))
            self.conv3 = GINConv(nn.Linear(128, out_dim))
        elif base_model == 'sage':
            self.conv1 = SAGEConv(in_dim, 256)
            self.conv2 = SAGEConv(256, 128)
            self.conv3 = SAGEConv(128, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = x / (x.sum())
        x = (F.tanh(x)) ** 2
        x = F.normalize(x)

        return x


def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj


def aux_objective(output, s):
    sample_size = len(s)

    out = output[s, :].float()

    C = oh_labels[s, :].float()

    X = C.sum(dim=0)
    X = X ** 2
    X = X.sum()

    Y = torch.matmul(torch.t(out), C)
    Y = torch.matmul(Y, torch.t(Y))
    Y = torch.trace(Y)

    t1 = torch.matmul(torch.t(C), C)
    t1 = torch.matmul(t1, t1)
    t1 = torch.trace(t1)

    t2 = torch.matmul(torch.t(out), out)
    t2 = torch.matmul(t2, t2)
    t2 = torch.trace(t2)

    t3 = torch.matmul(torch.t(out), C)
    t3 = torch.matmul(t3, torch.t(t3))
    t3 = torch.trace(t3)

    aux_objective_loss = 1 / (sample_size ** 2) * (t1 + t2 - 2 * t3)

    return aux_objective_loss


def regularization(output, s):
    out = output[s, :]
    ss = out.sum(dim=0)
    ss = ss ** 2
    ss = ss.sum()
    avg_sim = 1 / (len(s) ** 2) * ss

    return avg_sim ** 2


def loss_fn(output, lam=0.0, alp=0.0, epoch=-1):
    sample_size = int(1 * num_nodes)
    s = random.sample(range(0, num_nodes), sample_size)

    s_output = output[s, :]

    s_adj = sparse_adj[s, :][:, s]
    s_adj = convert_scipy_torch_sp(s_adj)
    s_degree = degree[s]

    x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))
    x = torch.matmul(x, s_output.double())
    x = torch.trace(x)

    y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
    y = (y ** 2).sum()
    y = y / (2 * num_edges)

    # scaling=1
    scaling = num_nodes ** 2 / (sample_size ** 2)

    m_loss = -((x - y) / (2 * num_edges)) * scaling

    aux_loss = lam * aux_objective(output, s)

    reg_loss = alp * regularization(output, s)

    loss = m_loss + aux_loss + reg_loss

    print('epoch: ', epoch, 'loss: ', loss.item(), 'm_loss: ', m_loss.item(), 'aux_loss: ', aux_loss.item(), 'reg_loss: ', reg_loss.item())

    return loss


def train(model, optimizer, data, epochs, lam, alp):
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)

        loss = loss_fn(out, lam, alp, epoch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    lam = args.lam
    alp = args.alp
    epochs = args.epochs
    device = args.device
    base_model = args.base_model
    seed = args.seed

    # if results exist then skip
    if alp == 0.0 and os.path.exists(f'results/results_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.pt'):
        print(f'results/results_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.pt exists. Skipping...')
        exit()
    elif alp != 0.0 and os.path.exists(f'results/results_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.pt'):
        print(f'results/results_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.pt exists. Skipping...')
        exit()

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # device selection
    if torch.cuda.is_available() and device != 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # transform data
    transform = T.NormalizeFeatures()

    # load dataset
    dataset = load_dataset(dataset_name)
    data = dataset[0]
    data = data.to(device)

    # preprocessing
    num_nodes = data.x.shape[0]
    num_edges = (data.edge_index.shape[1])
    labels = data.y.flatten()
    oh_labels = F.one_hot(labels, num_classes=max(labels) + 1)

    sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), data.edge_index.cpu().numpy()), shape=(num_nodes, num_nodes))
    torch_sparse_adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(num_edges).to(device), size=(num_nodes, num_nodes))
    degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(device)
    Graph = nx.from_scipy_sparse_array(sparse_adj, create_using=nx.Graph).to_undirected()
    num_edges = int((data.edge_index.shape[1]) / 2)

    in_dim = data.x.shape[1]
    out_dim = 64
    model = GNN(in_dim, out_dim, base_model=base_model).to(device)

    optimizer_name = "Adam"
    lr = 1e-3
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001, amsgrad=True)

    train(model, optimizer, data, epochs, lam, alp)

    test_data = data.clone()
    print(test_data)

    model.eval()
    x = model(test_data)

    clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(x.detach().cpu().numpy(), y=None)
    FQ = utils.compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device)
    print('No of clusters: ', max(clusters) + 1)
    print('Modularity:', FQ)

    NMI = utils.compute_nmi(clusters, data.y.squeeze().cpu().numpy())
    print('NMI:', NMI)

    conductance = utils.compute_conductance(clusters, Graph)
    avg_conductance = sum(conductance) / len(conductance)
    print(avg_conductance * 100)

    f1_score = utils.sample_f1_score(test_data, clusters, num_nodes)
    print('Sample_F1_score:', f1_score)

    results = {
        'num_clusters': np.unique(clusters).shape[0],
        'modularity': FQ,
        'nmi': NMI,
        'conductance': avg_conductance,
        'sample_f1_score': f1_score
    }

    if not os.path.exists('results'):
        os.makedirs('results')
    if alp == 0.0:
        torch.save(results, f'results/results_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.pt')
    else:
        torch.save(results, f'results/results_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.pt')
