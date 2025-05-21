### Environments

Install `bitsandbytes` on Jetson (version 0.41.2)

```
git clone https://github.com/to-aoki/bitsandbytes.git
cd bitsandbytes
(make sure right configurations for paths of nvcc, cuda, cuda-awared torch...)
CUDA_VERSION=114 make cuda11x
python setup.py install
Finish!
```