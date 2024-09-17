from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from tqdm import tqdm
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator

from util import smiles2graph
from preprocess import Chemical_feature_generator

# ['AlogP', 'Molecular_Weight', 'Num_H_Acceptors', 'Num_H_Donors',
#                  'Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea']

feature_label = ['MolWt', 'HeavyAtomMolWt',
                 'NumValenceElectrons', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
                 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                 'NumAliphaticRings', 'NumAromaticCarbocycles',
                 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'RingCount',
                 'MolMR', 'CalcNumBridgeheadAtom', 'ExactMolWt',
                 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
                 'NumSaturatedRings', 'MolLogP', 'CalcNumAmideBonds',
                 'CalcNumSpiroAtoms',
                 'num_ammonium_groups', 'num_alkoxy_groups']  # 29

# NumRadicalElectrons, num_carboxyl_groups, num_amion_groups, num_sulfonic_acid_groups --> train zero

given_features = ['AlogP', 'Molecular_Weight', 'Num_H_Acceptors', 'Num_H_Donors', 'Num_RotatableBonds', 'LogD',
                  'Molecular_PolarSurfaceArea']  # 7

generator = Chemical_feature_generator()


class ChemcialDataset(Dataset):
    def __init__(self,
                 data_frame: pd.DataFrame,
                 fps,
                 graphormer_data,
                 # mol_f,
                 transform=None,
                 is_train=True):
        super().__init__()
        self.df = data_frame
        self.fps = fps
        self.graphormer_data = graphormer_data
        # self.mol_f = mol_f
        self.transform = transform

        self.is_train = is_train

    def __getitem__(self, idx: IndexType | int):
        return self.get_chem_prop(idx)

    def __len__(self) -> int:
        return self.df.shape[0]

    def get_chem_prop(self, idx):

        sample = self.df.iloc[idx]
        fingerprint = self.fps[idx]
        smiles = sample['Smiles']

        # 'input_nodes', 'input_edges', 'attn_bias', 'in_degree', 'out_degree', 'spatial_pos', 'attn_edge_type'
        graphormer_data = self.graphormer_data[idx]

        edge_index, edge_attr = generator.get_adj_matrix(smiles=smiles)
        atomic_feature = generator.generate_mol_atomic_features(smiles=smiles)
        input_ids, attention_mask = generator.encoder_smiles(smiles)  # 384
        # ChemBERTa = ChemBERTa.detach()
        # molecular_feature = sample[feature_label] # if we use VarianceThreshold, then block this code

        atomic_feature = torch.tensor(atomic_feature, dtype=torch.float)
        fingerprint = torch.tensor(fingerprint, dtype=torch.float).view(1, -1)
        y = torch.tensor(sample['pIC50'])

        return Data(
            x=atomic_feature,
            fp=fingerprint,
            edge_index=edge_index,
            edge_attr=edge_attr,
            input_ids=input_ids,
            attention_mask=attention_mask,
            y=y,
            input_nodes=graphormer_data['input_nodes'],
            input_edges=graphormer_data['input_edges'],
            attn_bias=graphormer_data['attn_bias'],
            in_degree=graphormer_data['in_degree'],
            out_degree=graphormer_data['out_degree'],
            spatial_pos=graphormer_data['spatial_pos'],
            attn_edge_type=graphormer_data['attn_edge_type'],
        )


class KFoldDataModule(pl.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters(logger=False)

        self.train_data = None
        self.val_data = None
        self.num_cls = 0
        self.graphormer_collator = GraphormerDataCollator(on_the_fly_processing=True)

        self.setup()

    def setup(self, stage=None) -> None:
        if not self.train_data and not self.val_data:
            df = pd.read_csv(self.args.train_df, index_col=0)

            # mask = df['AlogP'] != df['AlogP']
            # df.loc[mask, 'AlogP'] = df.loc[mask, 'MolLogP']

            # if we use rdkit fingerprint generators
            # PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Molecule')
            # df["FPs"] = df.Molecule.apply(generator.get_molecule_fingerprints)
            # train_fps = np.stack(df["FPs"])
            mol2vec = []
            graphormer_data = []

            for smiles, pic50 in zip(df['Smiles'], df['pIC50']):
                vec = generator.get_mol_feature_from_deepchem(smiles=smiles)
                graph = smiles2graph(smiles)
                graph['labels'] = [pic50]
                graph = self.graphormer_collator([graph])

                mol2vec.append(vec)
                graphormer_data.append(graph)

            mol2vec = np.concatenate(mol2vec, axis=0)

            print('FPs feature shape:', mol2vec.shape)

            # fps = feature_selector.fit_transform(train_fps)

            kf = KFold(n_splits=self.args.num_split,
                       shuffle=True,
                       random_state=self.args.split_seed)
            all_splits = [k for k in kf.split(df)]
            train_idx, val_idx = all_splits[self.args.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train_df = df.iloc[train_idx]
            train_fp = mol2vec[train_idx]

            val_df = df.iloc[val_idx]
            val_fp = mol2vec[val_idx]

            self.train_data = ChemcialDataset(
                data_frame=train_df,
                fps=train_fp,
                graphormer_data=graphormer_data,
                transform=None,
                is_train=True
            )
            self.val_data = ChemcialDataset(
                data_frame=val_df,
                fps=val_fp,
                graphormer_data=graphormer_data,
                transform=None,
                is_train=True
            )

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          persistent_workers=self.args.persistent_workers,
                          pin_memory=self.args.pin_memory,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          persistent_workers=self.args.persistent_workers,
                          pin_memory=self.args.pin_memory)


# if __name__ == '__main__':
#     data = KFoldDataModule()
#
#     train_lodaer = data.train_dataloader()
#
#     for batch in train_lodaer:
#         # DataBatch(x=[29, 34], edge_index=[2, 62], edge_attr=[62], mol_f=[1, 36], fp=[5235], MLM=[1], HLM=[1], batch=[29], ptr=[2])
#         print(batch)
