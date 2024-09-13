import argparse

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
    if not test:
        df = df.sample(frac=1).reset_index(drop=True)

    # Calculate split index
    if test:
        split_ratio = 1.0

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
            pIC50 = [0]
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


def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    # SPLIT_RATIO 파라미터
    parser.add_argument('--split_ratio', type=float, default=0.95, help="Train-test split ratio")

    # LEARNING_RATE 파라미터
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for the optimizer")

    # EPOCH 파라미터
    parser.add_argument('--epoch', type=int, default=50, help="Number of epochs for training")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(f"SPLIT_RATIO: {args.split_ratio}")
    print(f"LEARNING_RATE: {args.learning_rate}")
    print(f"EPOCH: {args.epoch}")

    SPLIT_RATIO = args.split_ratio
    LEARNING_RATE = args.learning_rate
    EPOCH = args.epoch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda'
    print('using device:', device, torch.cuda.is_available())

    # Specify the path to the train.csv file
    csv_path = '../dataset/train.csv'
    test_csv_path = '../dataset/test.csv'

    # Process the CSV file and convert to graph format
    dataset = process_csv_to_graphs(csv_path, split_ratio=SPLIT_RATIO)
    test_dataset = process_csv_to_graphs(test_csv_path, split_ratio=SPLIT_RATIO, test=True)

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
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
        num_workers=6
    )

    validation_loader = DataLoader(
        dataset['validation'],
        batch_size=2,
        collate_fn=collator,
        num_workers=4
    )

    features = (
        'input_nodes', 'input_edges', 'attn_bias', 'in_degree', 'out_degree', 'spatial_pos', 'attn_edge_type', 'labels'
    )

    features_test = (
        'input_nodes', 'input_edges', 'attn_bias', 'in_degree', 'out_degree', 'spatial_pos', 'attn_edge_type'
    )


    class ThresholdPenaltyLoss(nn.Module):
        def __init__(self, threshold, penalty_weight):
            super(ThresholdPenaltyLoss, self).__init__()
            self.threshold = threshold  # 임계값
            self.penalty_weight = penalty_weight  # 벌점 가중치
            self.mse = nn.MSELoss()  # 기본 손실 함수 (MSE 사용)

        def forward(self, predictions, targets):
            # 기본 MSE 손실 계산
            mse_loss = self.mse(predictions, targets)

            # 임계값을 넘는 예측에 대해 벌점 부과
            over_threshold = torch.relu(predictions - self.threshold)  # 임계값을 넘는 부분
            penalty = self.penalty_weight * torch.sum(over_threshold)  # 넘는 부분에 대해 벌점 부과

            # 최종 손실 = MSE + 벌점
            total_loss = mse_loss + penalty

            return total_loss

    # criterion = ThresholdPenaltyLoss(
    #     threshold=0.5,
    #     penalty_weight=0.1
    # )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # for train in train_loader:
    #     print(train)

    def data_to_cuda(features, item):
        res = {}

        for feature in features:
            res[feature] = item[feature].to(device)

        return res

    # train
    for epoch in range(1, EPOCH + 1):
        losses = []

        errors = np.zeros(0)
        mses = np.zeros(0)
        predictions = np.zeros(0)

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
            abs_error_pic50 = np.abs(prediction.cpu().detach().numpy() - labels.cpu().detach().numpy()).squeeze(1)
            errors = np.concatenate((errors, abs_error_pic50))
            mses = np.concatenate((mses, abs_error_pic50 ** 2))
            predictions = np.concatenate((predictions, prediction.cpu().detach().numpy().squeeze(1)))

        A = np.mean(np.sqrt(mses) / (np.max(predictions) - np.min(predictions))).item()
        B = np.mean(errors <= 0.5).item()
        score = (0.5 * (1 - min(A, 1))) + 0.5 * B

        print('epoch {} loss {} score {} {:.2f} {:.2f}'.format(epoch, sum(losses) / len(losses), score, A, B))

        # validation
        if epoch % 5 == 0:
            model.eval()

            validation_losses = []
            errors = np.zeros(0)
            mses = np.zeros(0)
            predictions = np.zeros(0)

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

                abs_error_pic50 = np.abs(prediction.cpu().detach().numpy() - labels.cpu().detach().numpy()).squeeze(1)
                errors = np.concatenate((errors, abs_error_pic50))
                mses = np.concatenate((mses, abs_error_pic50 ** 2))
                predictions = np.concatenate((predictions, prediction.cpu().detach().numpy().squeeze(1)))

            A = np.mean(np.sqrt(mses) / (np.max(predictions) - np.min(predictions))).item()
            B = np.mean(errors <= 0.5).item()
            score = (0.5 * (1 - min(A, 1))) + 0.5 * B

            print('validation: {} score {}'.format(sum(validation_losses) / len(validation_losses), score))
    # eval

    # testing
    model.eval()  # Set the model to evaluation mode

    test_csv = pd.read_csv(test_csv_path)
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
    submission_df.to_csv("submission_{}_{}_{}.csv".format(EPOCH, LEARNING_RATE, SPLIT_RATIO), index=False)

    # submission_df2 = test_df[["ID", "IC50_nM", 'Smiles']]
    # submission_df2.to_csv("submission_{}_{}_{}_verify.csv".format(EPOCH, LEARNING_RATE, SPLIT_RATIO), index=False)

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
