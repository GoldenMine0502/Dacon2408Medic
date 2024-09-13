import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device = 'cuda'
# print('using device:', device, torch.cuda.is_available())

import pandas as pd
import numpy as np
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from datasets import Dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from transformers import GraphormerForGraphClassification
from transformers import TrainingArguments, Trainer
import os

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TORCH_USE_CUDA_DSA"] = '0'
os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(1)


def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


def smiles2graph(smiles_string, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
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

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_attr'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def process_csv_to_graphs(csv_path, split_ratio=0.8, test=False):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate split index
    split_index = int(len(df) * split_ratio)

    # Initialize lists to store graph data and labels for each split
    train_data = {'edge_index': [], 'edge_attr': [], 'node_feat': [], 'num_nodes': [], 'y': []}
    val_data = {'edge_index': [], 'edge_attr': [], 'node_feat': [], 'num_nodes': [], 'y': []}

    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        smiles = row['Smiles']

        if not test:
            pIC50 = row['pIC50']
        else:
            pIC50 = [0 for i in range(len(smiles))]
        # Convert SMILES to graph
        graph = smiles2graph(smiles)

        data_entry = {
            'edge_index': graph['edge_index'],
            'edge_attr': graph['edge_attr'],
            'node_feat': graph['node_feat'],
            'num_nodes': graph['num_nodes'],
            'y': [pIC50]
        }

        if idx < split_index:
            # Append to train data
            for key in train_data:
                train_data[key].append(data_entry[key])
        else:
            # Append to validation data
            for key in val_data:
                val_data[key].append(data_entry[key])

    # Create Dataset objects
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    # Create DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    return dataset


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda'
    print('using device:', device, torch.cuda.is_available())

    # Specify the path to the train.csv file
    csv_path = '../dataset/train.csv'
    test_csv_path = '../dataset/test.csv'

    # Process the CSV file and convert to graph format
    dataset = process_csv_to_graphs(csv_path, split_ratio=0.975)
    test_dataset = process_csv_to_graphs(test_csv_path, split_ratio=1, test=True)

    # Print out the dataset structure
    print(dataset)

    dataset_processed = dataset.map(preprocess_item, batched=False)
    model = GraphormerForGraphClassification.from_pretrained(
        "clefourrier/pcqm4mv2_graphormer_base",
        num_classes=1,  # num_classes for the downstream task
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # training_args = TrainingArguments(
    #     "graph-classification",
    #     logging_dir="graph-classification",
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     auto_find_batch_size=True,  # batch size can be changed automatically to prevent OOMs
    #     gradient_accumulation_steps=10,
    #     dataloader_num_workers=1,  # 1,
    #     num_train_epochs=20,
    #     evaluation_strategy="epoch",
    #     logging_strategy="epoch",
    #     push_to_hub=False,
    # )

    collator = GraphormerDataCollator(on_the_fly_processing=True)

    train_loader = DataLoader(
        dataset['train'],
        batch_size=24,
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )

    validation_loader = DataLoader(
        dataset['validation'],
        batch_size=24,
        collate_fn=collator,
        num_workers=4
    )


    features = (
        'input_nodes', 'input_edges', 'attn_bias', 'in_degree', 'out_degree', 'spatial_pos', 'attn_edge_type', 'labels'
    )


    features_test = (
        'input_nodes', 'input_edges', 'attn_bias', 'in_degree', 'out_degree', 'spatial_pos', 'attn_edge_type'
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    # for train in train_loader:
    #     print(train)

    def data_to_cuda(features, item):
        res = {}

        for feature in features:
            res[feature] = item[feature].to(device)

        return res

    # train
    EPOCH = 50
    for epoch in range(1, EPOCH + 1):
        losses = []

        model.train()
        for train in tqdm(train_loader, ncols=75):
            #         input_nodes: torch.LongTensor,
            #         input_edges: torch.LongTensor,
            #         attn_bias: torch.Tensor,
            #         in_degree: torch.LongTensor,
            #         out_degree: torch.LongTensor,
            #         spatial_pos: torch.LongTensor,
            #         attn_edge_type: torch.LongTensor,
            train = data_to_cuda(features, train)
            output = model(**train)
            prediction = output.logits
            labels = train['labels']
            # print(labels.dtype)
            prediction = prediction.type(labels.dtype)
            labels = labels.unsqueeze(1)
            # print(prediction.shape, labels.shape)
            loss = criterion(prediction, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss, loss.item())
            # print(loss.item())

            losses.append(loss.item())

        print('epoch {} loss {}'.format(epoch, sum(losses) / len(losses)))

        # validation
        if epoch % 5 == 0:
            model.eval()
            validation_losses = []

            for train in tqdm(validation_loader, ncols=75):
                #         input_nodes: torch.LongTensor,
                #         input_edges: torch.LongTensor,
                #         attn_bias: torch.Tensor,
                #         in_degree: torch.LongTensor,
                #         out_degree: torch.LongTensor,
                #         spatial_pos: torch.LongTensor,
                #         attn_edge_type: torch.LongTensor,
                train = data_to_cuda(features, train)
                output = model(**train)
                prediction = output.logits

                labels = train['labels']
                labels = labels.unsqueeze(1)

                prediction = prediction.type(labels.dtype)
                loss = criterion(prediction, labels)
                validation_losses.append(loss.item())

            print('validation: {}'.format(sum(validation_losses) / len(validation_losses)))
    # eval

    # testing
    model.eval()  # Set the model to evaluation mode

    test_loader = DataLoader(
        test_dataset['train'],
        batch_size=1,
        collate_fn=collator,
        num_workers=1
    )

    model.eval()
    test_predictions = []

    with torch.no_grad():
        for i, test in tqdm(enumerate(test_loader), ncols=75):
            test = data_to_cuda(features_test, test)
            output = model(**test)
            prediction = output.logits
            prediction = prediction.type(torch.float64)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            test_predictions.append(prediction.item())

    def pIC50_to_IC50(pic50_values):
        """Convert pIC50 values to IC50 (nM)."""
        return 10 ** (9 - pic50_values)

    print(test_predictions)
    print(np.array(test_predictions).shape)
    test_ic50_predictions = pIC50_to_IC50(np.array(test_predictions))

    # save results
    test_df = pd.read_csv('../dataset/test.csv')
    test_df["IC50_nM"] = test_ic50_predictions
    submission_df = test_df[["ID", "IC50_nM"]]
    submission_df.to_csv("submission.csv", index=False)


    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset_processed["train"],
    #     eval_dataset=dataset_processed["validation"],
    #     data_collator=GraphormerDataCollator(),
    # )
    # print(trainer.args.device)
    # train_results = trainer.train()
    # print(train_results)
    # trainer.push_to_hub()
