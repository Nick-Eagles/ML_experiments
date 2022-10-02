#   Create a basic environment with just python
conda create -y -p $PWD/ML1 python=3.8

#   Add packages we'll need
conda activate $PWD/ML1
python -m pip install tensorflow Pillow
conda deactivate
