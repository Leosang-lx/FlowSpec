### Environments

(Only for distributed multi-machine test)

Basic information

```
jetpack: 5.1.2
cuda: 11.4
python: 3.8
```

1. **pip install**

```shell
pip install -r requirements.txt
```

2. **Install torch and bitsandbytes separately**

`pip install torch-1.11.0-cp38-cp38-linux_aarch64.whl` (online resources)

Install `bitsandbytes` on Jetson (version 0.41.2)

```shell
git clone https://github.com/to-aoki/bitsandbytes.git

cd bitsandbytes

(make sure right configurations for paths of nvcc, cuda, cuda-awared torch...)

CUDA_VERSION=114 make cuda11x

python setup.py install

# Finish!
```



### Run

**First, get split models and configs** on server

Split model: `split_and_save_models.py`

- set proper model path to load model; set target path to save the *state_dict* of the split stage model
- set number of stages and layers
- run `python split_and_save_models.py`
- (Only for distributed test) send the *state_dict* of the split models and the weight of the draft models to the devices



**Then set configurations in**  `run_config.py`

- model name, model paths, running methods...



**After finish all above steps**, run `run_pipe.sh` (local test with multi-process) or `run_jetson.sh` (distributed test with multi-machine)

```
bash run_pipe.sh
# or
bash run_jetson.sh
```


