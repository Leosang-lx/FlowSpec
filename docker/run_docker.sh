# !/bin/bash

### build docker image
### docker build -t test_dict .

### run docker image
docker run -it \
  --gpus all \
  -v $HOME/big_file:/model_file \
  -v $HOME/project:/project \
  -u $(id -u):$(id -g) \
  -p 12345:12345 \
  --name lx_test \
  test_dist \
  bash
