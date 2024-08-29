import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# %%
# 학습 ChEMBL 데이터 로드
chembl_data = pd.read_csv('train.csv')  # 예시 파일 이름
print(f'Number of examples is: {len(chembl_data)}')
chembl_data.head()
# %%
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MLM"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Determine the maximum sequence length
max_length = tokenizer.model_max_length
print(max_length)


# %%
def tokenize(string):
    """
    Tokenize and encode a string using the provided tokenizer.

    Parameters:
        string (str): Input string to be tokenized.

    Returns:
        Tuple of input_ids and attention_mask.
    """
    encodings = tokenizer.encode_plus(
        string,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    return input_ids, attention_mask


# Tokenize the 'CANONICAL_SMILES' column and create new columns 'input_ids' and 'attention_mask'
tqdm.pandas()
chembl_data[["input_ids", "attention_mask"]] = chembl_data["Smiles"].progress_apply(lambda x: tokenize(x)).apply(
    pd.Series)
# %%
# Split the dataset into train, validation, and test sets
train_df, val_df = train_test_split(chembl_data, test_size=0.2, random_state=21)
print(f"There are {len(train_df)} molecules in Train df.")
print(f"There are {len(val_df)} molecules in Val df.")


# %%
# Function to convert data to PyTorch tensors
def get_tensor_data(data):
    """
    Convert data to PyTorch tensors.

    Parameters:
        data (DataFrame): Input data containing 'input_ids', 'attention_mask', and 'pIC50' columns.

    Returns:
        TensorDataset containing input_ids, attention_mask, and labels tensors.
    """
    input_ids_tensor = torch.tensor(data["input_ids"].tolist(), dtype=torch.int32)
    attention_mask_tensor = torch.tensor(data["attention_mask"].tolist(), dtype=torch.int32)
    labels_tensor = torch.tensor(data["pIC50"].tolist(), dtype=torch.float32)
    return TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)


# Create datasets and data loaders
train_dataset = get_tensor_data(train_df)
val_dataset = get_tensor_data(val_df)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
# %%
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Loss criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Decrease LR by a factor of 0.5 every 10 epochs
device = torch.device("cuda")
model.to(device)

epochs = 80
torch.manual_seed(12345)

for epoch in tqdm(range(epochs)):
    # Training loop
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        output_dict = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predictions = output_dict.logits.squeeze(dim=1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            output_dict = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = output_dict.logits.squeeze(dim=1)
            loss = criterion(predictions, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

    # Step the scheduler
    scheduler.step()


# %%
def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)


# %%
# Load the test data
test_df = pd.read_csv('test.csv')

# Tokenize the 'Smiles' column in the test dataset
tqdm.pandas()
test_df[["input_ids", "attention_mask"]] = test_df["Smiles"].progress_apply(lambda x: tokenize(x)).apply(pd.Series)


# %%
# Function to convert data to PyTorch tensors for the test set
def get_test_tensor_data(data):
    input_ids_tensor = torch.tensor(data["input_ids"].tolist(), dtype=torch.int32)
    attention_mask_tensor = torch.tensor(data["attention_mask"].tolist(), dtype=torch.int32)
    return TensorDataset(input_ids_tensor, attention_mask_tensor)


# Create test dataset and DataLoader
test_dataset = get_test_tensor_data(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# %%
# Testing loop
model.eval()  # Set the model to evaluation mode
test_predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output_dict = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = output_dict.logits.squeeze(dim=1)
        test_predictions.extend(predictions.tolist())

test_ic50_predictions = pIC50_to_IC50(np.array(test_predictions))
# %%
# Save the predictions to a submission file
test_df["IC50_nM"] = test_ic50_predictions
submission_df = test_df[["ID", "IC50_nM"]]
submission_df.to_csv("submission.csv", index=False)
