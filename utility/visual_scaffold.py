import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import cairosvg


def visualize_scaffolds(df, column='scaffold', n_mols=20, mols_per_row=5):
    """
    Visualize molecules from the 'Scaffold' column using RDKit without labels.

    Parameters:
        df (pd.DataFrame): Input dataframe containing SMILES strings.
        column (str): Column name containing SMILES strings (default: 'scaffold').
        n_mols (int): Number of molecules to visualize (default: 20).
        mols_per_row (int): Number of molecules per row in the image grid (default: 5).
    """
    # Select SMILES and convert to list
    smiles_list = df[column].dropna().unique()[:n_mols].tolist()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Filter out invalid SMILES
    mols = [mol for mol in mols if mol is not None]

    # Draw molecules without legends
    svg = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(200, 200), useSVG=True)
    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="./plots/interp_scaffold.pdf")


# the dataset will be the resulting dataset from the previous code
# that generates the top candidate scaffolds
df = pd.read_csv("./results/scaffold_permeability_whole.csv")
visualize_scaffolds(df)
