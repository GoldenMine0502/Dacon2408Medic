from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors


class ChemicalStructure:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.molecule = Chem.MolFromSmiles(smiles)

    def visualize(self):
        """
        Generates a visualization of the molecule.
        """
        if self.molecule:
            return Draw.MolToImage(self.molecule)
        else:
            raise ValueError("Invalid SMILES string.")

    def molecular_formula(self):
        """
        Returns the molecular formula of the molecule.
        """
        return Chem.rdMolDescriptors.CalcMolFormula(self.molecule)

    def molecular_weight(self):
        """
        Returns the molecular weight of the molecule.
        """
        return Chem.rdMolDescriptors.CalcExactMolWt(self.molecule)

    def is_valid(self):
        """
        Checks if the SMILES string is valid.
        """
        return self.molecule is not None


# Example SMILES strings
smiles_1 = "CN[C@@H](C)C(=O)N[C@H](C(=O)N1C[C@@H](NC(=O)CCOCCOCCOCC#Cc2cnc(OC[C@@H]3CCC(=O)N3)c3cc(OC)c(C(N)=O)cc23)C[C@H]1C(=O)N[C@@H]1CCCc2ccccc21)C1CCCCC1"
smiles_2 = "CC(C)(O)[C@H](F)CN1Cc2cc(NC(=O)c3cnn4cccnc34)c(N3CCC(N4CCC4)CC3)cc2C1=O"
smiles_3 = "COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc(C#CCCCCCCCCCCC(=O)N[C@H](C(=O)N3C[C@H](O)C[C@H]3C(=O)NCc3ccc(-c4scnc4C)cc3)C(C)(C)C)c2cc1C(N)=O"

# Create instances of the ChemicalStructure class
molecule_1 = ChemicalStructure(smiles_1)
molecule_2 = ChemicalStructure(smiles_2)
molecule_3 = ChemicalStructure(smiles_3)

# Visualize the molecules
img_1 = molecule_1.visualize()
img_2 = molecule_2.visualize()
img_3 = molecule_3.visualize()


# Print molecular formulas
print(molecule_1.molecular_formula())
print(molecule_2.molecular_formula())
print(molecule_3.molecular_formula())

# Print molecular weights
print(molecule_1.molecular_weight())
print(molecule_2.molecular_weight())
print(molecule_3.molecular_weight())

# Check if SMILES are valid
print(molecule_1.is_valid())
print(molecule_2.is_valid())
print(molecule_3.is_valid())


img_1.show()