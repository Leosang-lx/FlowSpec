#!/bin/sh

PACKAGE_NAME="huggingface-hub"

TOKEN=""
CACHE_DIR="/mnt/data1/big_file/liux/LLM/models_hf/"

while [ "$#" -gt 0 ]; do
    case $1 in
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --model)
            MODEL_TAG="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_TAG" ]; then
    echo "Please provide the model tag to download"
    exit 1
fi

CACHE_DIR="${CACHE_DIR}${MODEL_TAG}"

if [ -n "$TOKEN" ]; then
    echo "Provided HF token: $TOKEN"
fi

echo "Model: $MODEL_TAG"
echo "Cache-dir: $CACHE_DIR"

# 使用 pip show 来检查包是否已安装，并根据结果输出信息
if pip show $PACKAGE_NAME > /dev/null 2>&1; then
    echo "$PACKAGE_NAME is installed"
    export HF_ENDPOINT=https://hf-mirror.com
    echo "Start downloading..."
    if [ -n "$TOKEN" ]; then
        huggingface-cli download $MODEL_TAG --local-dir $CACHE_DIR --token $TOKEN
    else
        huggingface-cli download $MODEL_TAG --local-dir $CACHE_DIR
    fi
else
    echo "$PACKAGE_NAME not found, download cancelled"
    exit 1
fi