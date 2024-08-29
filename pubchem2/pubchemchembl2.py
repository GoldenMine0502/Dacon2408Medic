import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn

import torch.optim.lr_scheduler as lr_scheduler
from rdkit import Chem, DataStructs
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, \
    RobertaModel, RobertaTokenizer
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator

# 초기 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tqdm.pandas(ncols=75)
os.makedirs('chkpt', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# 데이터 로드
PRETRAIN_PATH = '../dataset/pubchem.chembl.dataset4publication_inchi_smiles.tsv'
PRETRAIN_FILTERED_PATH = '../dataset/filtered_pubchemchembl.tsv'

if not os.path.exists(PRETRAIN_FILTERED_PATH):
    print('caching data...')
    data = pd.read_csv(PRETRAIN_PATH, sep='\t')
    print('data loaded', len(data))

    # Activity_Flag가 'A'인 데이터만 필터링
    data = data[data['Activity_Flag'] == 'A']
    print('filtered data count:', len(data))

    # 잘못된 smile 데이터 제거 (smile에서 fingerprint 생성 불가 쳐냄)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


    def mol2fp(smile):
        # fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
        # fp = AllChem.GetMorganGenerator(mol, 2, nBits=4096)
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        fp = morgan_gen.GetFingerprint(mol)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)

        # 메모리 최적화
        del mol
        del fp

        return ar


    data["FPs"] = data["SMILES"].progress_apply(mol2fp)

    print('na count:', data.FPs.isna().sum())
    data = data[data['FPs'].notna()]
    print('final count:', len(data))

    data.to_csv(PRETRAIN_FILTERED_PATH, sep='\t', index=False)
else:
    # 미리 필터링된 파일 읽기 (읽는 속도 최적화)
    data = pd.read_csv(PRETRAIN_FILTERED_PATH, sep='\t')
    print('loaded cached data')


# 데이터셋 정의
class Dataset:
    def __init__(self, smiles, label, train=True):
        self.X = smiles
        self.y = label
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item_x = self.X[idx]
        item_y = self.y[idx]

        return item_x, item_y


VALIDATION_SPLIT = 0.05
validation_index = int((1 - VALIDATION_SPLIT) * len(data))
print('validation index, count:', validation_index, len(data))

train_smiles = data['SMILES'][:validation_index].values
train_labels = data['pXC50'][:validation_index].values
validation_smiles = data['SMILES'][validation_index:].values
validation_labels = data['pXC50'][validation_index:].values


def collate_fn(batch):
    x_list = []
    y_list = []
    for batch_X, batch_y in batch:
        x_list.append(batch_X)
        y_list.append(batch_y)

    return x_list, torch.tensor(y_list, dtype=torch.float32)


BATCH_SIZE = 140
train_loader = torch.utils.data.DataLoader(dataset=Dataset(train_smiles, train_labels),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader(dataset=Dataset(validation_smiles, validation_labels),
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                collate_fn=collate_fn)
print('data:', len(train_loader), len(validation_loader))

# 모델 로드
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model = RobertaForSequenceClassification(model.config)  # pretrain 안쓰고 학습
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model.to(DEVICE)
tokenizer.to(DEVICE)

max_length = tokenizer.model_max_length
print('max length:', max_length)
criterion = nn.MSELoss()
pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
pretrain_scheduler = lr_scheduler.StepLR(pretrain_optimizer, step_size=10, gamma=0.5)  # Decrease LR by a factor of 0.5 every 10 epochs

# pretrain
EPOCHS = 30


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


def train_and_validate(train_loader, validation_loader, optimizer, scheduler, epochs=EPOCHS):
    for epoch in range(1, epochs + 1):
        if train_loader is not None:
            model.train()
            total_train_loss = 0
            count = 0
            for (smiles, labels) in (pbar := tqdm(train_loader, ncols=75)):
                optimizer.zero_grad(set_to_none=True)

                inputs = tokenizer(smiles, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                labels = labels.to(DEVICE)

                output_dict = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = output_dict.logits.squeeze(dim=1)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                count += 1

                pbar.set_description(f'epoch: {epoch}, loss: {round(total_train_loss / count, 4)}')
            avg_train_loss = total_train_loss / count
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}")

            if epoch % 5 == 0:
                torch.save(model.state_dict(), f'chkpt/model_{epoch}.pt')

        if validation_loader is not None:
            # Validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for (smiles, labels) in tqdm(validation_loader, ncols=75):
                    inputs = tokenizer(smiles, return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']
                    labels = labels.to(DEVICE)

                    output_dict = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    predictions = output_dict.logits.squeeze(dim=1)
                    loss = criterion(predictions, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(validation_loader)

            print(f"Epoch {epoch}: Val Loss {avg_val_loss:.4f}")

        # Step the scheduler
        scheduler.step()


train_and_validate(train_loader, validation_loader, pretrain_optimizer, pretrain_scheduler)

# finetune
FINETUNE_PATH = '../dataset/train.csv'
finetune_data = pd.read_csv(FINETUNE_PATH)  # 예시 파일 이름
print(f'Number of finetune data is: {len(finetune_data)}')

finetune_train_smiles = finetune_data['Smiles']
finetune_train_labels = finetune_data['pIC50']

train_loader = torch.utils.data.DataLoader(dataset=Dataset(finetune_train_smiles, finetune_train_labels),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           collate_fn=collate_fn)


finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 기존 5e-5 -> 1e-5
finetune_scheduler = lr_scheduler.StepLR(pretrain_optimizer, step_size=10, gamma=0.5)  # Decrease LR by a factor of 0.5 every 10 epochs


train_and_validate(train_loader, None, finetune_optimizer, finetune_scheduler)  # validation 데이터가 딱히 없어서

# inference
TEST_PATH = '../dataset/test.csv'
test_data = pd.read_csv(TEST_PATH)

finetune_test_smiles = finetune_data['Smiles']
finetune_test_labels = np.zeros(len(test_data), dtype=float)

test_loader = torch.utils.data.DataLoader(dataset=Dataset(finetune_test_smiles, finetune_test_labels),
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          collate_fn=collate_fn)

model.eval()  # Set the model to evaluation mode
test_predictions = []

with torch.no_grad():
    for smiles, _ in test_loader:
        inputs = tokenizer(smiles, return_tensors='pt', padding=True).to(DEVICE)
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)

        output_dict = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = output_dict.logits.squeeze(dim=1)
        test_predictions.extend(predictions.tolist())


def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)


test_ic50_predictions = pIC50_to_IC50(np.array(test_predictions))

# Save the predictions to a submission file
test_data["IC50_nM"] = test_ic50_predictions
submission_df = test_data[["ID", "IC50_nM"]]
submission_df.to_csv("submission.csv", index=False)