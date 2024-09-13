#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


pip uninstall torch
pip install torch==2.3.1 torchaudio==2.3.1
TORCH=$(python -c 'import torch; print(torch.__version__)')
echo $TORCH
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# If you have installed dgl-cuXX package, please uninstall it first.
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

pip install wandb
pip install spacy
pip install lightning
pip install ogb


# install requirements
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb
pip install torch-geometric
pip install tensorboardX
pip install ogb
pip install rdkit-pypi
#pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html

cd fairseq
# if fairseq submodule has not been checkouted, run:
# git submodule update --init --recursive
pip install . --use-feature=in-tree-build
python setup.py build_ext --inplace

#pip install Cython

