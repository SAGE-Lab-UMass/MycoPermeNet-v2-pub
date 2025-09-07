import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split


# This function gets Murcko scaffold from a SMILES string
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    else:
        return None


df_mtb_1600 = pd.read_csv('./data/siegrist_clean_mtb1600.csv')
# Applying the scaffold extraction to the dataframe
df_mtb_1600['Scaffold'] = df_mtb_1600['Smiles'].apply(get_scaffold)

# Dropping entries where scaffold couldn't be determined
df_mtb_1600.dropna(subset=['Scaffold'], inplace=True)

# Grouping the dataset by scaffold
scaffold_groups = df_mtb_1600.groupby('Scaffold')

# Ensuring we split the scaffolds into train and test groups
scaffolds = list(scaffold_groups.groups.keys())
train_scaffolds, test_scaffolds = train_test_split(scaffolds, test_size=0.2, random_state=42)

# Ensuring to separate the dataframe into train and test sets based on scaffold groups
train_df = df_mtb_1600[df_mtb_1600['Scaffold'].isin(train_scaffolds)].reset_index(drop=True)
test_df = df_mtb_1600[df_mtb_1600['Scaffold'].isin(test_scaffolds)].reset_index(drop=True)

# Printing the results
print(f"Train set size: {train_df.shape[0]}")
print(f"Test set size: {test_df.shape[0]}")

# Saving as csv files
train_df.to_csv('./data/train_scaffold_split.csv')
test_df.to_csv('./data/test_scaffold_split.csv')
