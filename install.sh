conda install -c nvidia/label/cuda-11.3.1 cuda-toolkit
pip install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install pandas
pip install pymatgen
pip uninstall torch_spline_conv
pip install altair
pip install tensorboard
