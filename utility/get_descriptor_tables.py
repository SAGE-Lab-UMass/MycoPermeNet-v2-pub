import pandas as pd
import math


def generate_latex_table(df, title):
    cols = [c for c in df.columns if c.lower() != "smiles"]
    n = len(cols)

    rows = math.ceil(n / 5)

    latex = []
    latex.append(r"\begin{table}[!ht]")
    latex.append(r"\centering")
    latex.append(fr"\caption{{{title}}}")

    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{lllll}")
    latex.append(r"\toprule")

    latex.append(r"\multicolumn{5}{c}{Descriptor Name} \\")
    latex.append(r"\midrule")

    idx = 0
    for r in range(rows):
        row_entries = []
        for c in range(5):
            if idx < n:
                name = cols[idx].replace("_", r"\_")
                row_entries.append(rf"\text{{{name}}}")
            else:
                row_entries.append("")
            idx += 1
        latex.append(" & ".join(row_entries) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"}")  # end resizebox
    latex.append(r"\end{table}")

    return "\n".join(latex)


df1 = pd.read_csv("./data/preprocessed_labeled_descriptors.csv")
df2 = pd.read_csv("./data/preprocessed_labeled_descriptors_26.csv")

latex1 = generate_latex_table(df1, "Descriptor names from preprocessed\_labeled\_descriptors")
latex2 = generate_latex_table(df2, "Descriptor names from preprocessed\_labeled\_descriptors\_26")

print("====== TABLE 1 ======")
print(latex1)
print("\n====== TABLE 2 ======")
print(latex2)
