import time

import rdkit
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem, rdMolDescriptors, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.feature_selection import VarianceThreshold

import torch.nn as nn
import numpy as np
# import matplotlib as plt
import pandas as pd

# Ambit_InchiKey	Original_Entry_ID	Entrez_ID	Activity_Flag	pXC50	DB	Original_Assay_ID	Tax_ID	Gene_Symbol	Ortholog_Group	InChI	SMILES
# AAAAZQPHATYWOK-YRBRRWAQNA-N	11399331	2064	A	7.19382	pubchem	248914	9606	ERBB2	1346	InChI=1/C32H29ClN6O3S/c1-4-41-28-16-25-22(15-26(28)37-30(40)10-7-13-39(2)3)32(20(17-34)18-35-25)36-21-11-12-27(23(33)14-21)42-19-31-38-24-8-5-6-9-29(24)43-31/h5-12,14-16,18H,4,13,19H2,1-3H3,(H,35,36)(H,37,40)/b10-7+/f/h36-37H	ClC=1C=C(NC=2C=3C(N=CC2C#N)=CC(OCC)=C(NC(=O)/C=C/CN(C)C)C3)C=CC1OCC=4SC=5C(N4)=CC=CC5
# AAAAZQPHATYWOK-YRBRRWAQNA-N	CHEMBL175513	1956	A	6.73	chembl20	312997	9606	EGFR	1260	InChI=1/C32H29ClN6O3S/c1-4-41-28-16-25-22(15-26(28)37-30(40)10-7-13-39(2)3)32(20(17-34)18-35-25)36-21-11-12-27(23(33)14-21)42-19-31-38-24-8-5-6-9-29(24)43-31/h5-12,14-16,18H,4,13,19H2,1-3H3,(H,35,36)(H,37,40)/b10-7+/f/h36-37H	C1=2C(=C(C#N)C=NC1=CC(=C(C2)NC(/C=C/CN(C)C)=O)OCC)NC3=CC(Cl)=C(C=C3)OCC4=NC=5C=CC=CC5S4
# data = pd.read_csv("dataset/pubchem.chembl.dataset4publication_inchi_smiles.tsv", sep='\t')
# data.head(1)
# # Activity_Flag가 'A'인 데이터만 필터링
# filtered_df = data[data['Activity_Flag'] == 'A']
# # 필터링된 데이터를 새로운 tsv 파일로 저장
# filtered_df.to_csv('filtered.tsv', sep='\t', index=False)
current = time.time()
data = pd.read_csv('filtered.tsv', sep='\t')
print('data loaded', len(data), round(time.time() - current, 2))

PandasTools.AddMoleculeColumnToFrame(data, 'SMILES', 'Molecule')
data[["SMILES", "Molecule"]].head(1)
print('Molecule initialized', round(time.time() - current, 2))
# time.sleep(1)

print('na count:', data.Molecule.isna().sum())
data = data[data['Molecule'].notna()]
print('total:', len(data), round(time.time() - current, 2))
# time.sleep(1)

# data.drop(columns=['Molecule']).to_csv('filtered.tsv', sep='\t', index=False)

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
# morgan_gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=4096)


def mol2fp(mol):
    # fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    # fp = AllChem.GetMorganGenerator(mol, 2, nBits=4096)
    fp = morgan_gen.GetFingerprint(mol)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar


# fp = mol2fp(Chem.MolFromSmiles(data.loc[1, "SMILES"]))
# plt.matshow(fp.reshape((64, -1)), 0)

data["FPs"] = data.Molecule.apply(mol2fp)

X = np.stack(data.FPs.values)
print('X', X.shape)

y = data.pXC50.values.reshape((-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

#Normalizing output using standard scaling
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
y_validation = scaler.transform(y_validation)

# We'll remove low variance features
feature_select = VarianceThreshold(threshold=0.05)
X_train = feature_select.fit_transform(X_train)
X_validation = feature_select.transform(X_validation)
X_test = feature_select.transform(X_test)
print('train', X_train.shape)

# Let's get those arrays transfered to the GPU memory as tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# If you don't have a GPU, buy a graphics card. I have for a long time used a 1060 GTX, which is not that expensive anymore.
X_train = torch.tensor(X_train, device=device).float()
X_test = torch.tensor(X_test, device=device).float()
X_validation = torch.tensor(X_validation, device=device).float()
y_train = torch.tensor(y_train, device=device).float()
y_test = torch.tensor(y_test, device=device).float()
y_validation = torch.tensor(y_validation, device=device).float()
print(X_train)

train_dataset = TensorDataset(X_train, y_train)
validation_dataset = TensorDataset(X_validation, y_validation)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,
                                           shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=256,
                                                shuffle=False)
print('dataloader loaded')


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, out_size):
        super(Net, self).__init__()
        # Three layers and a output layer
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_size)  # Output layer
        # Layer normalization for faster training
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        # LeakyReLU will be used as the activation function
        self.activation = nn.LeakyReLU()
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):  # Forward pass: stacking each layer together
        # Fully connected =&amp;gt; Layer Norm =&amp;gt; LeakyReLU =&amp;gt; Dropout times 3
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.ln3(out)
        out = self.activation(out)
        out = self.dropout(out)
        # Final output layer
        out = self.fc_out(out)
        return out


# Defining the hyperparameters
input_size = X_train.size()[-1]  # The input size should fit our fingerprint size
hidden_size = 1024  # The size of the hidden layer
dropout_rate = 0.80  # The dropout rate
output_size = 1  # This is just a single task, so this will be one
learning_rate = 0.001  # The learning rate for the optimizer
model = Net(input_size, hidden_size, dropout_rate, output_size)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('training model')
model.train()  # Ensure the network is in "train" mode with dropouts active
epochs = 200
for e in range(epochs):
    running_loss = 0
    for fps, labels in train_loader:
        # Training pass
        optimizer.zero_grad()  # Initialize the gradients, which will be recorded during the forward pa

        output = model(fps)  # Forward pass of the mini-batch
        loss = criterion(output, labels)  # Computing the loss
        loss.backward()  # calculate the backward pass
        optimizer.step()  # Optimize the weights

        running_loss += loss.item()
    else:
        if e % 10 == 0:
            validation_loss = torch.mean((y_validation - model(X_validation)) ** 2).item()
            print("Epoch: %3i Training loss: %0.2F Validation loss: %0.2F" % (
                e, (running_loss / len(train_loader)), validation_loss))

model.eval()  #Swith to evaluation mode, where dropout is switched off
y_pred_train = model(X_train)
y_pred_validation = model(X_validation)
y_pred_test = model(X_test)

print('mean:', torch.mean((y_train - y_pred_train) ** 2).item())
print(torch.mean((y_validation - y_pred_validation) ** 2).item())
print(torch.mean((y_test - y_pred_test) ** 2).item())


# def flatten(tensor):
#     return tensor.cpu().detach().numpy().flatten()
#
#
# plt.scatter(flatten(y_pred_test), flatten(y_test), alpha=0.5, label="Test")
# plt.scatter(flatten(y_pred_train), flatten(y_train), alpha=0.1, label="Train")
# plt.legend()
# plt.plot([-1.5, 1.5], [-1.5,1.5], c="b")


def predict_smiles(smiles):
    fp = mol2fp(Chem.MolFromSmiles(smiles)).reshape(1, -1)
    fp_filtered = feature_select.transform(fp)
    fp_tensor = torch.tensor(fp_filtered, device=device).float()
    prediction = model(fp_tensor)
    #return prediction.cpu().detach().numpy()
    pXC50 = scaler.inverse_transform(prediction.cpu().detach().numpy())
    return pXC50[0][0]


print(predict_smiles('Cc1ccc2c(N3CCNCC3)cc(F)cc2n1'))
