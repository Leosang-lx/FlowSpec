# Gardener: Continuous Pipelined Speculative Decoding for Distributed LLM Inference
This repository is the official implementation of _Gardener: Continuous Pipelined Speculative Decoding for Distributed LLM Inference_

## Brief Introduction
In this work, we propose a pipeline-parallel tree-based speculative decoding framework for distributed inference, called **Gardener**, to reduce the inference latency with sparse requests. 
Our framework incorporates a lightweight draft model for token generation and the base LLM for pipeline-parallel verification, which enables the output of multiple tokens in a single forward propagation and hence mitigates the sparse request issue. 

## Requirements

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
bash run_eval.sh
# or
bash run_jetson_eval.sh
```

**13B model evaluation (Quantization is recommended)**
``` shell
# run
bash run_eval_13B.sh
# or
bash run_jetson_eval_13B.sh
```

## Pre-trained Models

We use the draft model weights provided by [EAGLE](https://github.com/SafeAILab/EAGLE/tree/main) for evaluation.

- [Vicuna-7B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3)
- [Vicuna-13B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-13B-v1.3)
- [LLaMA2-Chat 7B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-7B)
- [LLaMA2-Chat 13B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-13B)

## Results

Extended Results, which contain the performance comparison between Gardener and the
baselines across 6 datasets under two sampling settings (Temperature = 0 or 1, 0 means greedy
sampling). We select 20 samples for each dataset and limit the length of the generated sequences to
128. V and L2 are short for LLaMA2-Chat and Vicuna-v1.3, respectively. 7B and 13B denote the
number of parameters of the respective models.

| Model    | Method      | Metric       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean  | SR↑     |
|----------|-------------|--------------|---------:|----------:|------:|-------:|-------:|--------------:|------:|---------|
| **Temperature = 0** |  |  |  |  |  |  |  |  |  |  |
| V 13B    | Naive PP    | ξ ↑          |     0.98 |      1.23 |  1.20 |   1.06 |   0.62 |         0.81 | 0.98  | 1.00×   |
|          |             | Latency ↓    |   122.32 |    102.23 | 95.46 |  96.78 | 209.76 |       124.34 |125.15 |         |
|          | PipeDec     | ξ ↑          |     1.43 |      1.54 |  1.54 |   1.48 |   1.17 |         1.33 | 1.42  | 1.45×   |
|          |             | Latency ↓    |    81.54 |     80.52 | 72.46 |  67.98 | 110.17 |        73.18 | 80.98 |         |
|          | **Gardener**| **ξ ↑**      | **1.66** |  **2.05** |**2.02**|**1.80**|**1.09**|     **1.41** |**1.67**|**1.70×**|
|          |             | **Latency ↓**|**71.17** | **61.36** |**56.41**|**57.06**|**119.14**|**70.29**|**72.78**|         |
| V 7B     | Naive PP    | ξ ↑          |     6.58 |      5.29 |  6.30 |   5.95 |   3.46 |         4.07 | 5.28  | 1.00×   |
|          |             | Latency ↓    |    17.91 |     23.83 | 17.63 |  17.80 |  36.41 |        25.62 | 23.20 |         |
|          | PipeDec     | ξ ↑          |     6.99 |      6.71 |  7.19 |   6.95 |   5.14 |         5.88 | 6.48  | 1.23×   |
|          |             | Latency ↓    |    16.47 |     18.40 | 15.26 |  15.02 |  24.18 |        17.18 | 17.75 |         |
|          | **Gardener**| **ξ ↑**      | **8.93** |  **7.55** |**8.60**|**8.28**|**4.74**|**5.78**     |**7.31**|**1.38×**|
|          |             | **Latency ↓**|**12.85** | **16.61** |**12.84**|**12.77**|**26.41**|**17.91**|**16.57**|         |
| L2 13B   | Naive PP    | ξ ↑          |     1.40 |      1.60 |  1.35 |   1.22 |   1.09 |         1.11 | 1.30  | 1.00×   |
|          |             | Latency ↓    |    93.04 |     82.49 | 96.06 | 106.64 | 119.50 |       110.57 |101.38 |         |
|          | PipeDec     | ξ ↑          |     1.62 |      1.64 |  1.60 |   1.56 |   1.43 |         1.52 | 1.56  | 1.20×   |
|          |             | Latency ↓    |    78.35 |     78.44 | 78.25 |  82.23 |  90.01 |        80.97 | 81.38 |         |
|          | **Gardener**| **ξ ↑**      | **2.29** |  **2.69** |**2.26**|**2.04**|**1.81**|**1.89**     |**2.16**|**1.66×**|
|          |             | **Latency ↓**|**56.81** | **48.94** |**57.23**|**63.67**|**72.32**|**65.05**|**60.67**|         |
| L2 7B    | Naive PP    | ξ ↑          |     7.05 |      7.56 |  6.24 |   6.13 |   4.56 |         5.28 | 6.14  | 1.00×   |
|          |             | Latency ↓    |    18.38 |     17.48 | 20.97 |  21.10 |  28.92 |        21.55 | 21.40 |         |
|          | PipeDec     | ξ ↑          |     7.27 |      7.36 |  7.12 |   7.08 |   5.79 |         6.67 | 6.88  | 1.12×   |
|          |             | Latency ↓    |    17.50 |     17.52 | 18.08 |  17.79 |  22.28 |        16.78 | 18.33 |         |
|          | **Gardener**| **ξ ↑**      | **9.43** | **10.07** |**8.71**|**8.28**|**6.10**|**7.33**     |**8.32**|**1.35×**|
|          |             | **Latency ↓**|**13.60** | **13.02** |**14.96**|**15.31**|**21.48**|**15.44**|**15.64**|         |
| **Temperature = 1** |  |  |  |  |  |  |  |  |  |  |
| V 13B    | Naive PP    | ξ ↑          |     0.79 |      0.73 |  0.81 |   0.88 |   0.70 |         0.70 | 0.77  | 1.00×   |
|          |             | Latency ↓    |   118.54 |    158.61 |122.04 | 118.41 | 157.99 |       154.48 |138.34 |         |
|          | PipeDec     | ξ ↑          |     1.31 |      1.37 |  1.26 |   1.40 |   1.21 |         1.34 | 1.32  | 1.71×   |
|          |             | Latency ↓    |    88.21 |     82.39 | 87.55 |  70.92 | 103.27 |        78.10 | 85.07 |         |
|          | **Gardener**| **ξ ↑**      | **1.38** |  **1.36** |**1.36**|**1.47**|**1.17**|**1.26**     |**1.33**|**1.73×**|
|          |             | **Latency ↓**|**69.58** | **76.65** |**81.00**|**71.88**|**104.82**|**85.02**|**81.50**|         |
| V 7B     | Naive PP    | ξ ↑          |     4.12 |      4.15 |  3.71 |   4.56 |   3.13 |         3.47 | 3.86  | 1.00×   |
|          |             | Latency ↓    |    28.13 |     28.16 | 29.07 |  22.47 |  38.00 |        31.94 | 29.63 |         |
|          | PipeDec     | ξ ↑          |     6.19 |      6.34 |  5.92 |   6.28 |   5.12 |         5.64 | 5.92  | 1.53×   |
|          |             | Latency ↓    |    19.45 |     19.83 | 18.24 |  14.86 |  23.51 |        19.33 | 19.20 |         |
|          | **Gardener**| **ξ ↑**      | **5.76** |  **5.77** |  **5.47**|**6.17**|**4.63**|**5.16**|**5.49**|**1.42×**|
|          |             | **Latency ↓**|**20.10** | **18.85** |**18.47**|**16.82**|**27.05**|**19.17**|**20.08**|         |
| L2 13B   | Naive PP    | ξ ↑          |     1.36 |      1.51 |  1.22 |   1.20 |   1.02 |         1.15 | 1.24  | 1.00×   |
|          |             | Latency ↓    |    94.70 |     87.78 |106.12 | 106.30 | 126.84 |       110.32 |105.34 |         |
|          | PipeDec     | ξ ↑          |     1.61 |      1.66 |  1.60 |   1.55 |   1.45 |         1.55 | 1.57  | 1.27×   |
|          |             | Latency ↓    |    78.51 |     77.65 | 80.29 |  80.00 |  88.96 |        81.95 | 81.23 |         |
|          | **Gardener**| **ξ ↑**      | **2.18** |  **2.29** |  **2.09**|**1.99**|**1.67**|**1.82**|**2.01**|**1.62×**|
|          |             | **Latency ↓**|**58.90** | **57.33** |**62.42**|**62.53**|**77.96**|**68.68**|**64.64**|         |
| L2 7B    | Naive PP    | ξ ↑          |     6.46 |      6.48 |  5.95 |   5.87 |   4.17 |         5.04 | 5.66  | 1.00×   |
|          |             | Latency ↓    |    19.72 |     18.01 | 22.21 |  21.64 |  31.37 |        23.53 | 22.75 |         |
|          | PipeDec     | ξ ↑          |     7.11 |      7.06 |  7.25 |   7.14 |   5.78 |         6.82 | 6.86  | 1.21×   |
|          |             | Latency ↓    |    17.30 |     17.52 | 17.80 |  16.87 |  22.32 |        17.85 | 18.28 |         |
|          | **Gardener**| **ξ ↑**      | **8.53** |  **8.63** |**8.43**|**7.54**|**5.79**|**7.00**|**7.65**|**1.35×**|
|          |             | **Latency ↓**|**14.84** | **14.58** |**15.38**|**16.86**|**22.24**|**16.43**|**16.72**|         |
---

## Acknowledgement
The implementation of Gardener reuses the code from [EAGLE](https://github.com/SafeAILab/EAGLE) and refers to [OPT-Tree](https://github.com/Jikai0Wang/OPT-Tree) and [Jupiter](https://github.com/ysyisyourbrother/Jupiter).

## Contributing
coming soon...

