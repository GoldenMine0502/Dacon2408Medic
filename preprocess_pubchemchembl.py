import time
import numpy as np

import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tqdm.pandas(ncols=75)

# Ambit_InchiKey	Original_Entry_ID	Entrez_ID	Activity_Flag	pXC50	DB	Original_Assay_ID	Tax_ID	Gene_Symbol	Ortholog_Group	InChI	SMILES
# AAAAZQPHATYWOK-YRBRRWAQNA-N	11399331	2064	A	7.19382	pubchem	248914	9606	ERBB2	1346	InChI=1/C32H29ClN6O3S/c1-4-41-28-16-25-22(15-26(28)37-30(40)10-7-13-39(2)3)32(20(17-34)18-35-25)36-21-11-12-27(23(33)14-21)42-19-31-38-24-8-5-6-9-29(24)43-31/h5-12,14-16,18H,4,13,19H2,1-3H3,(H,35,36)(H,37,40)/b10-7+/f/h36-37H	ClC=1C=C(NC=2C=3C(N=CC2C#N)=CC(OCC)=C(NC(=O)/C=C/CN(C)C)C3)C=CC1OCC=4SC=5C(N4)=CC=CC5
# AAAAZQPHATYWOK-YRBRRWAQNA-N	CHEMBL175513	1956	A	6.73	chembl20	312997	9606	EGFR	1260	InChI=1/C32H29ClN6O3S/c1-4-41-28-16-25-22(15-26(28)37-30(40)10-7-13-39(2)3)32(20(17-34)18-35-25)36-21-11-12-27(23(33)14-21)42-19-31-38-24-8-5-6-9-29(24)43-31/h5-12,14-16,18H,4,13,19H2,1-3H3,(H,35,36)(H,37,40)/b10-7+/f/h36-37H	C1=2C(=C(C#N)C=NC1=CC(=C(C2)NC(/C=C/CN(C)C)=O)OCC)NC3=CC(Cl)=C(C=C3)OCC4=NC=5C=CC=CC5S4
current = time.time()
data = pd.read_csv("dataset/pubchem.chembl.dataset4publication_inchi_smiles.tsv", sep='\t')
# data = pd.read_csv('filtered.tsv', sep='\t')
print('data loaded', len(data), round(time.time() - current, 2))

# # Activity_Flag가 'A'인 데이터만 필터링
data = data[data['Activity_Flag'] == 'A']

# PandasTools.AddMoleculeColumnToFrame(data, 'SMILES', 'Molecule')
# data[["SMILES", "Molecule"]].head(1)
# print('Molecule initialized', round(time.time() - current, 2))
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
print('total:', len(data), round(time.time() - current, 2))

X = np.stack(data.FPs.values)
y = data.pXC50.values.reshape((-1, 1))
print('train data:', X.shape)
print('label data:', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
print('len (x train, x test, y train, y test, x_vali, y_vali):', len(X_train), len(X_test), len(y_train), len(y_test),
      len(X_validation), len(y_validation))
# Normalizing output using standard scaling
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
y_validation = scaler.transform(y_validation)

# We'll remove low variance features
feature_select = VarianceThreshold(threshold=0.05)
X_train = feature_select.fit_transform(X_train)
X_validation = feature_select.transform(X_validation)
X_test = feature_select.transform(X_test)
print('shape (fit):', X_train.shape)

np.save("dataset/pubchemchembl_x_train.npy", X_train)
np.save("dataset/pubchemchembl_x_vali.npy", X_validation)
np.save("dataset/pubchemchembl_x_test.npy", X_test)
np.save("dataset/pubchemchembl_y_train.npy", y_train)
np.save("dataset/pubchemchembl_y_vali.npy", y_validation)
np.save("dataset/pubchemchembl_y_test.npy", y_test)
data.to_csv('dataset/filtered_pubchemchembl.tsv', sep='\t', index=False)
