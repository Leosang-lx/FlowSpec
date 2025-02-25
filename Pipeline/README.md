# About Pipeline Directory

Distributed Communication and Computation Device
- backend: `gloo`
- device
  - `cpu`
  - `gpu`
    - 1. `tensor.cpu()`
    - 2. `dist.comm(tensor)`

Explanation for file

`stage_ea_model.py`: the modeling python file for modeling the partial base model for pipeline stages on different devices.

- refer  to implemenation of jupiter

`modeling_llama_kv_eagle.py`: the modeling python file of pipeline stage for a certain kind of base model ("llama" here)

- other base model should have a file name like `modeling_[base_model_name]_kv_eagle.py`

