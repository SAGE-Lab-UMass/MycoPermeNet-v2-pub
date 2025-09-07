import pandas as pd
from rdkit import Chem

# 40
sieg_40 = pd.read_excel('./data/20240813_comprehensive_data(2).xlsx',
                         sheet_name = 'smiles_react_perm_40')
df_mtb_40 = pd.DataFrame({
    'Smiles': sieg_40['Smile'],
    'MTB Standardized Residuals': sieg_40['mtb_resid_std']
})

# 380
sieg_380 = pd.read_excel('./data/20240813_comprehensive_data(2).xlsx',
                         sheet_name = 'smiles_react_perm_380')
df_mtb_380 = pd.DataFrame({
    'Smiles': sieg_380['Smile'],
    'MTB Standardized Residuals': sieg_380['mtb_resid_std']
})

# 1200
sieg_1200 = pd.read_excel('./data/20240813_comprehensive_data(2).xlsx',
                         sheet_name = 'smiles_react_perm_1200')
df_mtb_1200 = pd.DataFrame({
    'Smiles': sieg_1200['Smile'],
    'MTB Standardized Residuals': sieg_1200['mtb_resid_std']
})

# 1600
df_mtb_1600  = pd.concat([df_mtb_1200, df_mtb_380, df_mtb_40], axis=0)

df_mtb_1600.dropna(axis=0, inplace=True)
print('Dataset shape after dropping NaN:', df_mtb_1600.shape)

# Filter invalid SMILES strings
valid_smiles = []
invalid_indices = []
for i, smile in enumerate(df_mtb_1600['Smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        valid_smiles.append(smile)
    else:
        invalid_indices.append(i)

# Remove invalid entries from df
df_mtb_1600 = df_mtb_1600.drop(invalid_indices).reset_index(drop=True)

print('\nDataset shape after dropping invalid SMILES strings:', df_mtb_1600.shape)

# Save the cleaned dataset CSV file
df_mtb_1600.to_csv('./data/siegrist_clean_mtb1600.csv', index=False)
print('\nDataset saved successfully!')
