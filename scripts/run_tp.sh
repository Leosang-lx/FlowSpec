MASTER_ADDR=${MASTER_ADDR:-"master.node.address"}    # leader 节点地址
MASTER_PORT=${MASTER_PORT:-29500}                    # leader 节点端口
NUM_NODES=${NUM_NODES:-2}                            # 节点总数
GPUS_PER_NODE=${GPUS_PER_NODE:-4}                    # 每节点 GPU 数
NODE_RANK=${NODE_RANK:-0}                            # 本节点排行 (0~NUM_NODES-1)
HOSTFILE=${HOSTFILE:-"./hostfile"}                  # 主机文件路径
DS_CONFIG=${DS_CONFIG:-"./ds_config.json"}           # DeepSpeed 配置文件

# ———— 二、DeepSpeed Launcher 方式 ————
deepspeed \
  --hostfile ${HOSTFILE} \
  --num_nodes ${NUM_NODES} \
  --num_gpus ${GPUS_PER_NODE} \
  --no_ssh \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  train.py \
  --deepspeed \
  --deepspeed_config ${DS_CONFIG} \
  "$@"