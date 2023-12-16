# Search Beyond Queries: Grounding Language Agents for Web Interactions via Reinforcement Learning

In this repository, the code for our paper "Search Beyond Queries: Grounding Language Agents for Web
Interactions via Reinforcement Learning" is provided

## Installation steps

1. **Clone the project**
```
git clone https://github.com/MultifacetedNLP/Web-Agents-Unsupervised.git Web-Agents-Unsupervised; cd Web-Agents-Unsupervised
```
2. **Create conda env**
```
conda create -n WebAgent python=3.10.8; conda activate WebAgent
```
3. **Install PyTorch**
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
4. **Install required packages**
```
pip install -r requirements.txt
```
5. **Install Wevshop Environment**: 
```
cd web_agent_site; pip install -e .; cd ..
```
6. **If the environment variable $SCRATCH (storing datsets and LLM models) is empty, set it up**
```
export SCRATCH="/path/to/your/desired/folder/scratch_folder"
```
7. **Download the Datasets and Prepare the Search Engine for Environment**
```
chmod +x ./setup.sh; ./setup.sh -d all
```
8. **Install Accelerate**
```
cd v0.13.2/accelerate-0.13.2; pip install -e .; cd ../..
```
9. **Install Lamorel**
```
cd lamorel/lamorel; pip install -e .; cd ../..
```

## Train
The following code will trian the Flan-T5 large model in the webshop environment using Proximal Policy Optimization.
```

```

## Inference
The following code will download the model that was only trained with PPO, and put it in the $SCATCH path
```
cd $SCRATCH;
mkdir -p storage/models;
cd storage/models;
gdown "https://drive.google.com/uc?id=1GYumAWzrIyo-fby5wT5JsXjkto-8bbzq&confirm=t";
unzip flan_t5_large_2_observations_only_ppo_1000000_steps.zip
```
Run the following code for inference on the only-ppo model
```
```