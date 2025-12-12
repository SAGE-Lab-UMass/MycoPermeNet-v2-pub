import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import cairosvg


def visualize_scaffolds(df, column='scaffold', value_col='predicted_permeability', n_mols=20, mols_per_row=5):
    """
    Visualize molecules from the 'Scaffold' column using RDKit without labels.

    Parameters:
        df (pd.DataFrame): Input dataframe containing SMILES strings.
        column (str): Column name containing SMILES strings (default: 'scaffold').
        n_mols (int): Number of molecules to visualize (default: 20).
        mols_per_row (int): Number of molecules per row in the image grid (default: 5).
    """
    # Select unique scaffolds
    sub_df = df[[column, value_col]].dropna().drop_duplicates(subset=[column]).head(n_mols)

    smiles_list = sub_df[column].tolist()
    values_list = sub_df[value_col].tolist()

    # Convert SMILES to RDKit molecules
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]

    # Create legends (two decimal places)
    legends = [f"{v:.2f}" for v in values_list]

    # Draw molecules with legends
    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(200, 200),
        useSVG=True,
        legends=legends
    )
    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="./plots/interp_scaffold.pdf")


# the dataset will be the resulting dataset from the previous code
# that generates the top candidate scaffolds
df = pd.read_csv("./results/scaffold_permeability_whole.csv")
visualize_scaffolds(df)
