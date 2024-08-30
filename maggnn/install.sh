echo "venv를 만든 상태에서 실행해주세요. torch==2.3.1, cu12.1이 설치됩니다."
echo "python3 -m venv venv"
echo "source venv/bin/activate"
sleep 10

pip uninstall torch
pip install torch==2.3.1
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

python -m spacy download en