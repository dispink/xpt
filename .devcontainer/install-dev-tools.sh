pip3 install --upgrade --user pip; 
pip3 --no-cache-dir install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118;
pip3 --no-cache-dir install --user -r .devcontainer/requirements.txt;