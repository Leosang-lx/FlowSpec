# for MODEL_CONFIG AND NETWORK_CONFIG
import ipaddress
import datetime
from types import SimpleNamespace
from socket import AF_INET, AF_INET6
import os
import torch.cuda
import torch.distributed as dist

# from autoregressive_inference import config, model_config
distributed = False
world_size = 2

timeout_max = datetime.timedelta(seconds=10)


def get_device_and_distributed_backend(use_gpu=False):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    backend = 'gloo'  # only use gloo for comm
    if torch.cuda.is_available() and use_gpu:  # GPU is available and choose GPU
        return 'cuda', backend
    return 'cpu', backend
    #     device = 'cuda'
    #     if dist.is_nccl_available():  # prefer NCCL if available
    #         backend = 'nccl'
    #     elif dist.is_mpi_available():
    #         backend = 'mpi'
    #     else:
    #         raise RuntimeError('Not available distributed backend for GPU comm')
    # else:  # CPU
    #     device = 'cpu'
    #     backend = 'gloo'
    # return device, backend


def get_network_config(is_distributed=False, use_gpu=False):
    MASTER_IP = '192.168.1.106'
    INTERFACE = 'eth0'
    SUBNET = '192.168.1'
    PORT_MASTER = 9999
    # PORT_TCP = 8848
    PORT_DISTRIBUTED = 23456

    network_config = {  # these configurations work for both local and distributed
        'master_addr': (MASTER_IP, PORT_MASTER),
        'interface': INTERFACE,
        'subnet': SUBNET,
        'port_distributed': PORT_DISTRIBUTED,
    }
    if is_distributed:
        device, backend = get_device_and_distributed_backend(use_gpu)
        # init ip_rank_mapping
        rank_suffix = [161, 162, 163, 164]
        rank_ip_mapping = [f'{SUBNET}.{suffix}' for suffix in rank_suffix]
        world_size = len(rank_suffix)
        ipvx = AF_INET
        # Enable when using RaspberryPi
        if backend == 'gloo':
            os.environ['GLOO_SOCKET_IFNAME'] = INTERFACE
            os.environ['GLOO_TIMEOUT'] = '20s'
        elif backend == 'mpi':
            os.environ['MPI_SOCKET_IFNAME'] = INTERFACE
            os.environ['MPI_TIMEOUT'] = '20s'
        else:
            # nccl暂时无法使用
            raise RuntimeError('Unsupported backend for torch.distributed')
    else:
        device, backend = get_device_and_distributed_backend(
            False)  # use cpu by default for local distributed inference
        rank_ip_mapping = ['::1']
        world_size = 2
        # MAIN_WORKER_IP = '::1'
        # ipvx = AF_INET6  # test with multi-process in Windows: fail to use ipv4
        pass

    network_config['device'] = device
    network_config['backend'] = backend
    network_config['rank_ip_mapping'] = rank_ip_mapping
    network_config['world_size'] = world_size

    return SimpleNamespace(**network_config)


def ipv4_or_ipv6(ip):
    try:
        ipaddress.IPv4Address(ip)
        return 4
    except ipaddress.AddressValueError:
        pass

    try:
        ipaddress.IPv6Address(ip)
        return 6
    except ipaddress.AddressValueError:
        pass

    return 0


def gen_init_method(rank0_ip: str, port: int, protocol='tcp'):
    ipv = ipv4_or_ipv6(rank0_ip)
    if ipv == 4:
        init_method = f'{protocol}://{rank0_ip}:{port}'
    elif ipv == 6:
        init_method = f'{protocol}://[{rank0_ip}]:{port}'
    else:
        raise Exception('Invalid IP Address')

    return init_method

# def get_simple_model_config():
#     return model_config
