import pandas as pd

PRETRAIN_FILTERED_PATH = '../dataset/filtered_pubchemchembl.tsv'
TEST_PATH = '../dataset/test.csv'

data = pd.read_csv(PRETRAIN_FILTERED_PATH, sep='\t')
test_data = pd.read_csv(TEST_PATH)

pretrain_smiles = data['SMILES']
finetune_test_smiles = test_data['Smiles']

print(len(pretrain_smiles.max()), len(finetune_test_smiles.max()))