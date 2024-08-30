import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, LayerNorm
from torch.optim import lr_scheduler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from tqdm import tqdm


# 데이터 전처리: SMILES -> 그래프
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    bonds = []
    bond_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.append(bond.GetBondTypeAsDouble())
        bond_indices.append((i, j))

    # torch_geometric의 Data 객체 생성
    data = Data(
        x=torch.tensor(atoms, dtype=torch.float).view(-1, 1),
        edge_index=torch.tensor(bond_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(bonds, dtype=torch.float)
    )

    return data

# GNN 모델 정의
class GNNModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, dropout_rate=0.8):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, 256)
        self.fc2 = torch.nn.Linear(256, input_size)
        self.ln1 = LayerNorm(hidden_size)  # Batch Normalization 레이어
        self.ln2 = LayerNorm(hidden_size)  # Batch Normalization 레이어
        self.ln3 = LayerNorm(hidden_size)  # Batch Normalization 레이어
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        out = self.conv1(x, edge_index)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out, edge_index)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv3(out, edge_index)
        out = self.ln3(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = global_mean_pool(out, data.batch)  # 그래프 풀링

        # Final output layer
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# 모델 초기화 및 예측
model = GNNModel()

data = pd.read_csv('../dataset/train.csv')
print(f'Number of examples is: {len(data)}')
print(data[['Smiles', 'pIC50']].head())

VALIDATION_SPLIT = 0.05
validation_index = int((1 - VALIDATION_SPLIT) * len(data))
print('validation index, count:', validation_index, len(data))

train_smiles = data['Smiles'][:validation_index].values
train_labels = data['pIC50'][:validation_index].values
validation_smiles = data['Smiles'][validation_index:].values
validation_labels = data['pIC50'][validation_index:].values

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Decrease LR by a factor of 0.5 every 10 epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


epochs = 3
# torch.manual_seed(12345)

for epoch in tqdm(range(epochs)):
    # Training loop
    model.train()
    total_train_loss = 0
    for smile, label in zip(train_smiles, train_labels):
        optimizer.zero_grad(set_to_none=True)

        label = torch.tensor(float(label)).to(device)
        graph = smiles_to_graph(smile).to(device)

        output = model(graph)
        # output = output.item()
        loss = criterion(label, output)

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_smiles)

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for smile, label in zip(validation_smiles, validation_labels):
            optimizer.zero_grad(set_to_none=True)

            label = torch.tensor(float(label)).to(device)
            graph = smiles_to_graph(smile).to(device)

            output = model(graph)
            loss = criterion(label, output)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(validation_smiles)

    print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

    # Step the scheduler
    scheduler.step()


# testing
test_df = pd.read_csv('../dataset/test.csv')
test_smiles = test_df['Smiles']

model.eval()  # Set the model to evaluation mode
test_predictions = []

with torch.no_grad():
    for smile in test_smiles:
        graph = smiles_to_graph(smile).to(device)
        output = model(graph)
        output = output.item()
        test_predictions.append(output)


def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)


test_ic50_predictions = pIC50_to_IC50(np.array(test_predictions))

test_df["IC50_nM"] = test_ic50_predictions
submission_df = test_df[["ID", "IC50_nM"]]
submission_df.to_csv("submission.csv", index=False)
