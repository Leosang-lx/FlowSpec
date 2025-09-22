import os

# os.environ['GLOO_USE_IPV6'] = '0'
# os.environ['TP_USE_IPV6'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'
os.environ['MASTER_ADDR'] = '172.18.36.132'
os.environ['MASTER_PORT'] = '12345'

rank = int(os.environ['RANK'])
masterIP = os.environ['MASTER_ADDR']
masterPort = os.environ['MASTER_PORT']

import torch.distributed as dist
print(f'Process {os.getpid()} initialize with rank {rank} with master {f"{masterIP}:{masterPort}"}...')
try:
    dist.init_process_group(init_method='env://', backend='gloo')
    print('success')
except:
    print('fail')
    import traceback
    traceback.print_exc()
finally:
    if dist.is_initialized():
        dist.destroy_process_group()


# check listening port
# windows: Get-NetTCPConnection -LocalPort 12345 -State Listen
# linux: netstat -nlp | grep 12345