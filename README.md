# Search Beyond Queries: Grounding Language Agents for Web Interactions via Reinforcement Learning

In this repository, the code for our paper "Search Beyond Queries: Grounding Language Agents for Web
Interactions via Reinforcement Learning" is provided

## Installation steps
1. **Create conda env**
```
conda create -n dlp python=3.10.8; conda activate dlp
```
2. **Install PyTorch**
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
3. **Install packages required by our package**
```
pip install -r requirements.txt
```
4. **Install Wevshop Environment**: 
```
here
```
6. **Install Accelerate**
```
cd v0.13.2/accelerate-0.13.2; pip install -e .; cd ../..
```
7. **Install Lamorel**
```
cd lamorel/lamorel; pip install -e .; cd ../..
```