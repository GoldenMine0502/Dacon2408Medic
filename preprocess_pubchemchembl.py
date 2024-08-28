import time

import pandas as pd
from rdkit.Chem import PandasTools

# Ambit_InchiKey	Original_Entry_ID	Entrez_ID	Activity_Flag	pXC50	DB	Original_Assay_ID	Tax_ID	Gene_Symbol	Ortholog_Group	InChI	SMILES
# AAAAZQPHATYWOK-YRBRRWAQNA-N	11399331	2064	A	7.19382	pubchem	248914	9606	ERBB2	1346	InChI=1/C32H29ClN6O3S/c1-4-41-28-16-25-22(15-26(28)37-30(40)10-7-13-39(2)3)32(20(17-34)18-35-25)36-21-11-12-27(23(33)14-21)42-19-31-38-24-8-5-6-9-29(24)43-31/h5-12,14-16,18H,4,13,19H2,1-3H3,(H,35,36)(H,37,40)/b10-7+/f/h36-37H	ClC=1C=C(NC=2C=3C(N=CC2C#N)=CC(OCC)=C(NC(=O)/C=C/CN(C)C)C3)C=CC1OCC=4SC=5C(N4)=CC=CC5
# AAAAZQPHATYWOK-YRBRRWAQNA-N	CHEMBL175513	1956	A	6.73	chembl20	312997	9606	EGFR	1260	InChI=1/C32H29ClN6O3S/c1-4-41-28-16-25-22(15-26(28)37-30(40)10-7-13-39(2)3)32(20(17-34)18-35-25)36-21-11-12-27(23(33)14-21)42-19-31-38-24-8-5-6-9-29(24)43-31/h5-12,14-16,18H,4,13,19H2,1-3H3,(H,35,36)(H,37,40)/b10-7+/f/h36-37H	C1=2C(=C(C#N)C=NC1=CC(=C(C2)NC(/C=C/CN(C)C)=O)OCC)NC3=CC(Cl)=C(C=C3)OCC4=NC=5C=CC=CC5S4

current = time.time()
data = pd.read_csv("dataset/pubchem.chembl.dataset4publication_inchi_smiles.tsv", sep='\t')
# data = pd.read_csv('filtered.tsv', sep='\t')
print('data loaded', len(data), round(time.time() - current, 2))

# # Activity_Flag가 'A'인 데이터만 필터링
data = data[data['Activity_Flag'] == 'A']

PandasTools.AddMoleculeColumnToFrame(data, 'SMILES', 'Molecule')
data[["SMILES", "Molecule"]].head(1)
print('Molecule initialized', round(time.time() - current, 2))

print('na count:', data.Molecule.isna().sum())
data = data[data['Molecule'].notna()]
print('total:', len(data), round(time.time() - current, 2))

data.drop(columns=['Molecule']).to_csv('filtered_pubchemchembl.tsv', sep='\t', index=False)