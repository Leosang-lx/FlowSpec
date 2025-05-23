# Gardener: Continuous Pipelined Speculative Decoding for Distributed LLM Inference

## Brief Introduction
In this work, we propose a pipeline-parallel tree-based speculative decoding framework for distributed inference, called **Gardener**, to reduce the inference latency with sparse requests. 
Our framework incorporates a lightweight draft model for token generation and the base LLM for pipeline-parallel verification, which enables the output of multiple tokens in a single forward propagation and hence mitigates the sparse request issue. 

## Environments Setup

### Basic information

```
jetpack: 5.1.2
cuda: 11.4
python: 3.8
```

### Virtual Environment Setup

```shell
python -m venv gardener

source ~/venv/gardener/bin/activate

pip install -r requirements.txt

# Or for Jetson
pip install -r requirements_jetson.txt
```

**Or Conda Environment Setup**

```shell
conda create -n gardener python=3.8

conda activate gardener

pip install -r requirements.txt

# Or for Jetson
pip install -r requirements_jetson.txt
```

**Install Torch And Bitsandbytes (Only for Jetson)**

```shell
# Download the torch-1.11.0-cp38-cp38-linux_aarch64.whl from online resources
wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl

pip install torch-1.11.0-cp38-cp38-linux_aarch64.whl

# Install bitsandbytes for Jetson (version 0.41.2)
git clone https://github.com/to-aoki/bitsandbytes.git

cd bitsandbytes

(make sure right configurations for paths of nvcc, cuda, cuda-awared torch...)

CUDA_VERSION=114 make cuda11x

python setup.py install
```

## Model Weights

We use the draft model weights provided by [EAGLE](https://github.com/SafeAILab/EAGLE/tree/main) for evaluation.

- [Vicuna-7B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3)
- [Vicuna-13B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-13B-v1.3)
- [LLaMA2-Chat 7B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-7B)
- [LLaMA2-Chat 13B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-13B)

## Quick Start

**First, get split models and configs** on server

Split model: `split_and_save_models.py`

- set proper model path to load model; set target path to save the *state_dict* of the split stage model
- set number of stages and layers
- run `python split_and_save_models.py`
- (Only for distributed test) send the *state_dict* of the split models and the weight of the draft models to the devices

**Then set configurations in**  `config/run_config.py`

- model name, model paths, running methods...

**Finished all above steps**, run `run_pipe.sh` (local test with multi-process) or `run_jetson.sh` (distributed test with multi-machine)

``` shell
# split models and save
python split_and_save_models.py

# set configurations for running
sudo nano config/run_config.py

# run
bash run_pipe.sh
# or
bash run_jetson.sh
```

## Evaluation

To start large scale evaluation, run `run_pipe_eval.sh` or `run_jetson_eval.sh` for 7B model for local and distributed scenarios, respectively. 

Set `quant` in `run_config.py` to choose the quantization method, if needed.

**7B model evaluation**
``` shell
# run
bash run_pipe_eval.sh
# or
bash run_jetson_eval.sh
```

**13B model evaluation (Quantization is recommended)**
``` shell
# run
bash run_pipe_eval_13B.sh
# or
bash run_jetson_eval_13B.sh
```

