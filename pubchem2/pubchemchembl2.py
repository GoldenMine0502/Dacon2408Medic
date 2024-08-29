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

PRETRAIN_PATH = '../dataset/pubchem.chembl.dataset4publication_inchi_smiles.tsv'
PRETRAIN_FILTERED_PATH = '../dataset/filtered_pubchemchembl.tsv'
FINETUNE_PATH = '../dataset/train.csv'
TEST_PATH = '../dataset/test.csv'
VALIDATION_SPLIT = 0.05

# MODEL_MAX_LEN = 256
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 5e-5
LEARNING_RATE_FINETUNE = 1e-3
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(BATCH_SIZE, EPOCHS, LEARNING_RATE, LEARNING_RATE_FINETUNE, DEVICE)


# 데이터 로드

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

        self.token = list(map(lambda x: tokenize(x), self.X))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item_x = self.X[idx]
        item_y = self.y[idx]
        token = self.token[idx]

        return item_x, item_y, token


validation_index = int((1 - VALIDATION_SPLIT) * len(data))
print('validation index, count:', validation_index, len(data))

train_smiles = data['SMILES'][:validation_index].values
train_labels = data['pXC50'][:validation_index].values
validation_smiles = data['SMILES'][validation_index:].values
validation_labels = data['pXC50'][validation_index:].values


def collate_fn(batch):
    x_list = []
    y_list = []
    tokens = []
    for batch_X, batch_y, token in batch:
        x_list.append(batch_X)
        y_list.append(batch_y)
        tokens.append(token)

    # token = tokenizer(x_list, return_tensors='pt', padding=True)

    return x_list, torch.tensor(y_list, dtype=torch.float32), tokens


train_loader = torch.utils.data.DataLoader(dataset=Dataset(train_smiles, train_labels),
                                           batch_size=BATCH_SIZE,
                                           # num_workers=6,
                                           shuffle=True,
                                           collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader(dataset=Dataset(validation_smiles, validation_labels),
                                                batch_size=BATCH_SIZE,
                                                # num_workers=6,
                                                shuffle=False,
                                                collate_fn=collate_fn)
print('data:', len(train_loader), len(validation_loader))

# 모델 로드
pretrained_model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

# max_length = pretrained_model.config.max_position_embeddings
# print('max length:', max_length)
# pretrained_model.config.max_position_embeddings = MODEL_MAX_LEN
# print('max length set to:', MODEL_MAX_LEN)

model = RobertaForSequenceClassification(pretrained_model.config)  # pretrain 안쓰고 학습
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
max_length = tokenizer.model_max_length
model.to(DEVICE)

del pretrained_model

criterion = nn.MSELoss()
pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
pretrain_scheduler = lr_scheduler.StepLR(pretrain_optimizer, step_size=10, gamma=0.5)  # Decrease LR by a factor of 0.5 every 10 epochs


# pretrain
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


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def IC50_to_pIC50(ic50_values):
    """Convert IC50 values (nM) to pIC50."""
    return 9 - (np.log10(ic50_values))


def train_and_validate(train_loader, validation_loader, optimizer, scheduler, epochs=EPOCHS):
    for epoch in range(1, epochs + 1):
        if train_loader is not None:
            model.train()
            total_train_loss = 0
            count = 0
            for smiles, labels, token in (pbar := tqdm(train_loader, ncols=75)):
                optimizer.zero_grad(set_to_none=True)

                input_ids = token['input_ids'].to(DEVICE)
                attention_mask = token['attention_mask'].to(DEVICE)
                labels = labels.to(DEVICE)

                output_dict = model(input_ids=input_ids, attention_mask=attention_mask)
                prediction = output_dict.logits.squeeze(dim=1)
                loss = criterion(prediction, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                count += 1

                pbar.set_description(f'loss: {total_train_loss / count:.4f}')
            avg_train_loss = total_train_loss / count
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}")

            if epoch % 5 == 0:
                torch.save(model.state_dict(), f'chkpt/model_{epoch}.pt')

        if validation_loader is not None:
            # Validation loop
            model.eval()
            total_val_loss = 0
            errors = np.zeros(0)
            mses = np.zeros(0)
            predictions = np.zeros(0)
            with torch.no_grad():
                for smiles, labels, token in tqdm(validation_loader, ncols=75):
                    input_ids = token['input_ids'].to(DEVICE)
                    attention_mask = token['attention_mask'].to(DEVICE)
                    labels = labels.to(DEVICE)

                    output_dict = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    prediction = output_dict.logits.squeeze(dim=1)

                    loss = criterion(prediction, labels)
                    total_val_loss += loss.item()

                    abs_error_pic50 = np.abs(prediction.cpu().detach().numpy() - labels.cpu().detach().numpy())
                    errors = np.concatenate((errors, abs_error_pic50))
                    mses = np.concatenate((mses, abs_error_pic50 ** 2))
                    predictions = np.concatenate((predictions, prediction.cpu().detach().numpy()))
            avg_val_loss = total_val_loss / len(validation_loader)

            A = np.mean(np.sqrt(mses) / (np.max(predictions) - np.min(predictions))).item()
            B = np.mean(abs_error_pic50 <= 0.5).item()

            print(f"Epoch {epoch}: Val Loss {avg_val_loss:.4f} accuracy: {A:.2f} {B:.2f} {(0.5 * (1 - min(A, 1))) + 0.5 * B}")

        # Step the scheduler
        scheduler.step()


train_and_validate(train_loader, validation_loader, pretrain_optimizer, pretrain_scheduler)

# finetune
finetune_data = pd.read_csv(FINETUNE_PATH)  # 예시 파일 이름
print(f'Number of finetune data is: {len(finetune_data)}')

finetune_train_smiles = finetune_data['Smiles']
finetune_train_labels = finetune_data['pIC50']

train_loader = torch.utils.data.DataLoader(dataset=Dataset(finetune_train_smiles, finetune_train_labels),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           collate_fn=collate_fn)


finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE)  # 기존 5e-5 -> 1e-5
finetune_scheduler = lr_scheduler.StepLR(pretrain_optimizer, step_size=10, gamma=0.5)  # Decrease LR by a factor of 0.5 every 10 epochs


train_and_validate(train_loader, None, finetune_optimizer, finetune_scheduler)  # validation 데이터가 딱히 없어서

# inference
test_data = pd.read_csv(TEST_PATH)

finetune_test_smiles = test_data['Smiles']
finetune_test_labels = np.zeros(len(test_data), dtype=float)

test_loader = torch.utils.data.DataLoader(dataset=Dataset(finetune_test_smiles, finetune_test_labels),
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          collate_fn=collate_fn)

model.eval()  # Set the model to evaluation mode
test_predictions = []

with torch.no_grad():
    for smiles, _, token in test_loader:
        input_ids = token['input_ids'].to(DEVICE)
        attention_mask = token['attention_mask'].to(DEVICE)

        output_dict = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = output_dict.logits.squeeze(dim=1)
        test_predictions.extend(prediction.tolist())


def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)


test_ic50_predictions = pIC50_to_IC50(np.array(test_predictions))

# Save the predictions to a submission file
test_data["IC50_nM"] = test_ic50_predictions
submission_df = test_data[["ID", "IC50_nM"]]
submission_df.to_csv("submission.csv", index=False)