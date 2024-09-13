import random

import numpy as np
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GINConv, GATConv, GCNConv, JumpingKnowledge, global_max_pool, global_mean_pool
from dgllife.utils import *
from rdkit import Chem
from torch_geometric.data import Data, Batch
from Transformer import Transformer


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    8 features are canonical, 2 features are from OGB
    """
    featurizer_funcs = ConcatFeaturizer([atom_type_one_hot,
                                         atom_degree_one_hot,
                                         atom_implicit_valence_one_hot,
                                         atom_formal_charge,
                                         atom_num_radical_electrons,
                                         atom_hybridization_one_hot,
                                         atom_is_aromatic,
                                         atom_total_num_H_one_hot,
                                         atom_is_in_ring,
                                         atom_chirality_type_one_hot,
                                         ])
    atom_feature = featurizer_funcs(atom)
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    featurizer_funcs = ConcatFeaturizer([bond_type_one_hot,
                                         # bond_is_conjugated,
                                         # bond_is_in_ring,
                                         # bond_stereo_one_hot,
                                         ])
    bond_feature = featurizer_funcs(bond)

    return bond_feature


def smiles2graph(mol):
    """
    Converts SMILES string or rdkit's mol object to graph Data object without remove salt
    :input: SMILES string (str)
    :return: graph object
    """

    if isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        mol = Chem.MolFromSmiles(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    #     num_bond_features = 3  # bond type, bond stereo, is_conjugated
    num_bond_features = 1  # bond type
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = Data(x=torch.tensor(x, dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr), dtype=torch.float)

    return graph


class DrugEncoder(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()

        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, x, edge_index, batch=None):
        # x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_drug = 3
        self.dim_drug = 128
        self.dim_drug_cell = 256  # 256 to 128
        self.dropout_ratio = 0.1

        self.DrugEncoder = DrugEncoder(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, self.dim_drug_cell),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio)
        )

        self.Transformer = Transformer(d_model=self.dim_drug_cell, nhead=8, num_encoder_layers=1,
                                       dim_feedforward=self.dim_drug_cell)
        reg_input = self.dim_drug_cell  # *2를 빼야함
        reg_hidden = 512

        self.regression = nn.Sequential(
            nn.Linear(reg_input, reg_hidden),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(reg_hidden, reg_hidden),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(reg_hidden, 1)
        )

    def forward(self, x, edge_index, batch=None):
        # ---- (1) forward drug ----
        x = self.DrugEncoder(x, edge_index, batch)
        x = self.drug_emb(x)

        # ---- (3) combine drug feature and cell line feature ----
        x = self.regression(x)

        return x


class GAT_GCN_Transformer(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=77, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GAT_GCN_Transformer, self).__init__()

        self.n_output = n_output
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=num_features_xd, nhead=1, dropout=0.2)
        self.ugformer_layer_1 = nn.TransformerEncoder(self.encoder_layer_1, 1)
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=num_features_xd * 10, nhead=1, dropout=0.2)
        self.ugformer_layer_2 = nn.TransformerEncoder(self.encoder_layer_2, 1)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        # self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        # self.pool_xt_1 = nn.MaxPool1d(3)
        # self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        # self.pool_xt_2 = nn.MaxPool1d(3)
        # self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        # self.pool_xt_3 = nn.MaxPool1d(3)
        # self.fc1_xt = nn.Linear(2944, output_dim)

        # combined layers
        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch=None):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x.shape)
        x = torch.unsqueeze(x, 1)
        x = self.ugformer_layer_1(x)
        x = torch.squeeze(x, 1)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.ugformer_layer_2(x)
        x = torch.squeeze(x, 1)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        # # protein input feed-forward:
        # target = data.target_mut
        # target = target[:, None, :]
        # # 1d conv layers
        # conv_xt = self.conv_xt_1(target)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = self.pool_xt_1(conv_xt)
        # conv_xt = self.conv_xt_2(conv_xt)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = self.pool_xt_2(conv_xt)
        # conv_xt = self.conv_xt_3(conv_xt)
        # conv_xt = F.relu(conv_xt)
        # conv_xt = self.pool_xt_3(conv_xt)
        #
        # # flatten
        # xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        # xt = self.fc1_xt(xt)
        #
        # # concat
        # xc = torch.cat((x, xt), 1)
        xc = x
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        # out = nn.Sigmoid()(out)
        return out


def collate_fn(samples):
    samples = list(samples)
    random.shuffle(samples)

    drugs, labels = map(list, zip(*samples))

    # indices = [idx for idx in range(0, len(samples))]
    # drugs = drugs[indices]
    # labels = labels[indices]

    batched_drug = Batch.from_data_list(drugs)
    # batched_cell = Batch.from_data_list(cells)
    return batched_drug, torch.tensor(labels)


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


from SmilesEnumerator import SmilesEnumerator
sme = SmilesEnumerator()


def augmentation(smiles, labels):
    result_smiles = []
    result_labels = []
    for smile, label in tqdm(zip(smiles, labels)):
        for i in range(10):
            result_smiles.append(sme.randomize_smiles(smile))
            result_labels.append(label)

    return result_smiles, result_labels


def main():
    # dataset
    train_df = pd.read_csv('../dataset/train.csv')  # 예시 파일 이름
    test_df = pd.read_csv('../dataset/test.csv')

    train_smile = train_df['Smiles'].to_list()
    train_label = train_df['pIC50'].to_list()
    train_smile, train_label = augmentation(train_smile, train_label)
    train_dataset = [(smiles2graph(mol), torch.tensor([label])) for mol, label in tqdm(zip(train_smile, train_label))]

    test_smile = test_df['Smiles'].to_list()
    test_dataset = [smiles2graph(mol) for mol in test_smile]

    train_loader = DataLoader(
        MyDataset(train_dataset),
        batch_size=10,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # test_loader = DataLoader(
    #     MyDataset(test_dataset),
    #     batch_size=1,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     num_workers=4
    # )

    # loading model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GAT_GCN_Transformer()
    model.to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    EPOCH = 60
    train_losses = []
    for epoch in range(1, EPOCH + 1):
        # train
        model.train()

        random.shuffle(train_dataset)
        for graph, label in tqdm(train_loader, ncols=75):
            label = label.to(device).view(-1, 1)

            output = model(
                x=graph.x.to(device),
                edge_index=graph.edge_index.to(device),
                # batch=graph.batch.to(device),
            )
            print(output.shape, label.shape)
            loss = criterion(output, label)
            # print(round(output.item(), 2), round(label.item(), 2), round(loss.item(), 2))
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        print(f'epoch: {epoch}, loss: {sum(train_losses) / len(train_losses)}')

    # test
    results = []
    model.eval()
    for graph in tqdm(test_dataset, ncols=75):
        # graph = graph[0]
        output = model(
            x=graph.x.to(device),
            edge_index=graph.edge_index.to(device)
        )
        results.append(output.item())

    test_ic50_predictions = 10 ** (9 - np.array(results))

    # Save the predictions to a submission file
    test_df["IC50_nM"] = test_ic50_predictions
    submission_df = test_df[["ID", "IC50_nM"]]
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
