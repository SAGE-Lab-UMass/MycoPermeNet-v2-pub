# MycoPermeNet-v2

[![Environment](https://github.com/SAGE-Lab-UMass/MycoPermeNet-v2-pub/actions/workflows/evaluate.yml/badge.svg)](https://github.com/SAGE-Lab-UMass/MycoPermeNet-v2-pub/actions/workflows/evaluate.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/SAGE-Lab-UMass/MycoPermeNet-v2-pub/blob/main/LICENSE)

***Improved Prediction of Mycomembrane Permeation Using Fusion Noisy Student Self-distillation***

## Directory structure

The directory structure of this project is shown as below:

- [.github/workflows/](.github/workflows/) contains the yml test files for GitHub Actions.

- [data](data/) stores critical raw data, train_val and test permeability data, preprocessed labeled descriptors, unlabeled datasets for NST. The preprocessed large MLSMR dataset is provided in this [Google Drive](https://drive.google.com/drive/folders/1qY9JcMwK-HUQ2g2xuNPPnVxtyEXt4jxh?usp=sharing), and all the [MoleculeNet benchmark datasets](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.MoleculeNet.html) are from the [PyG library](https://pyg.org/).

- [data_tools](data_tools/) has scripts to compute descriptors, preprocess data, construct datasets, etc.

- [model](model/) contains the pretrained Chemprop and MLP checkpoints of the MycoPermeNet-v2 model under the best random state.

- [models](models/) defines the GNN encoders, MINE model, and MLP.

- [results](results/) will save the results files.

- [script](script/) has the scripts to run MycoPermeNet-v2 and all the other experiments in the paper.

- [utility](utility) has some helper scripts to visualize results and get LaTex tables.

## Installation

To set up the environment and install dependencies for this project, follow below steps: 

1. Clone the repository

```bash
git clone https://github.com/SAGE-Lab-UMass/MycoPermeNet-v2-pub.git
cd MycoPermeNet-v2-pub
```

2. Create and activate a virtual environment

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train and test

Our MycoPermeNet-v2 pipeline is implemented based on the [PyG framework](https://pyg.org/). Given a custom dataset or dataset from MoleculeNet, the pipeline can automatically preprocess the data, split data, construct data loaders, train two-stage models with selected GNN encoders and proposed Fusion/NST strategies, and finally test the model.

To train MycoPermeNet-v2 for [permeability property prediction](script/mycoperme_nst.py), use the command line:

```bash
python script/mycoperme_nst.py --GNN chemprop --fusion True --NST True --NST_volume 500
```

Train MycoPermeNet-v2 for other [properties' prediction from MoleculeNet](script/moleculenet_nst.py), use the command line:

```bash
python script/moleculenet_nst.py --moldataset Lipo --GNN AttentiveFP --fusion True --NST True --NST_volume 1000
```
