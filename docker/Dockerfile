# Dockerfile
FROM docker.1ms.run/nvidia/cuda:12.8.0-devel-ubuntu22.04
# FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip nano sudo iperf3
RUN pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# COPY 

WORKDIR /app