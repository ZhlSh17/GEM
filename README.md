# GEM

## 💾 Installation
```bash
conda create -n gem python=3.10
conda activate gem

git clone --recurse-submodules https://github.com/ZhlSh17/GEM.git
cd GEM

pip install -r requirements.txt

# [Optional] install flash attention; can be slow to build
pip install flash-attn --no-build-isolation
```
