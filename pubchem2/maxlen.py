import pandas as pd

TEST_PATH = '../dataset/test.csv'
test_data = pd.read_csv(TEST_PATH)

finetune_test_smiles = test_data['Smiles']

print(len(finetune_test_smiles.max()))