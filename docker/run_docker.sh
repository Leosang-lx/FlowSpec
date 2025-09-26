# !/bin/bash

### build docker image
### docker build -t test_dist .

### continue
### docker exec -it contain_name bash

### run docker image
#   -u $(id -u):$(id -g) \
#   -p 12345:12345 \
docker run -it \
  --gpus all \
  -v $HOME/big_file:/model_file \
  -v $HOME/project:/project \
  --name lx_test \
  --network host \
  test_dist \
  bash
