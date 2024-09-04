import numpy as np
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import tqdm
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool
from dgllife.utils import *
from rdkit import Chem
from torch_geometric.data import Data
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


def main():
    # dataset
    train_df = pd.read_csv('../dataset/train.csv')  # 예시 파일 이름
    test_df = pd.read_csv('../dataset/test.csv')

    train_smile = train_df['Smiles'].to_list()
    train_label = train_df['pIC50'].to_list()
    train_dataset = [(smiles2graph(mol), torch.tensor([label])) for mol, label in zip(train_smile, train_label)]

    test_smile = test_df['Smiles'].to_list()
    test_dataset = [smiles2graph(mol) for mol in test_smile]

    # loading model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyModel()
    model.to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []

    EPOCH = 60
    for epoch in range(1, EPOCH + 1):
        # train
        model.train()
        for graph, label in tqdm(train_dataset, ncols=75):
            label = label.to(device).unsqueeze(0)

            output = model(
                x=graph.x.to(device),
                edge_index=graph.edge_index.to(device)
            )

            loss = criterion(output, label.float())
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        print(f'epoch: {epoch}, loss: {sum(train_losses) / len(train_losses)}')

    # test
    results = []
    model.eval()
    for graph in tqdm(test_dataset, ncols=75):
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
