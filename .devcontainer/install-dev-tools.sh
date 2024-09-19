pip3 install --upgrade --user pip; 
pip3 --no-cache-dir install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118;
#pip3 --no-cache-dir install --user -r .devcontainer/requirements.txt;

# # For jupyter notebook
pip3 --no-cache-dir install --user ipython ipykernel ipywidgets tqdm;

# For ML
pip3 --no-cache-dir install --user \
    transformers \
    timm tensorboard\
    scikit-learn \
    tensorboard \
    torch-tb-profiler;

# For data science
pip3 --no-cache-dir install --user \
    scipy \
    numpy \
    pandas \
    openpyxl \
    matplotlib \
    JPype1 \
    tabula-py;

# Dependency issues
pip3 --no-cache-dir install --user protobuf==4.25