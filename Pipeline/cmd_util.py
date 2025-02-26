from subprocess import getstatusoutput


# def get_ip_addr(__subnet__: str):  # get ip by subnet
def get_ip_addr(network_interface: str):  # get ip by interface or subnet
    status, ip_addr = getstatusoutput(f"ifconfig {network_interface} | grep netmask | awk '{{print $2}}'")
    # status, ip_addr = getstatusoutput(f"ifconfig | grep {__subnet__} | awk '{{print $2}}'")
    if status == 0:
        return ip_addr
    else:
        raise Exception('The interface has no ip')


# Deprecated
# def cpulimit(__pid__: int, new_cpu_limit: int):  # 这玩意是启动一个监控线程一直控制给定线程的利用率，比较不方便
#     run(['sudo', 'cpulimit', '-p', str(__pid__), '-l', str(new_cpu_limit)])


def get_available_cpu_frequency():
    status, output = getstatusoutput(f'cpupower frequency-info | grep "available frequency steps"')
    if status == 0:
        frequencies = output.split(';')[1].strip()
        frequencies = [frequency.strip() for frequency in frequencies.split(',')]
        return frequencies
    return None


def cpu_frequency_limit(__frequency__: str):
    cmd = f'sudo cpupower -c all frequency-set -u {__frequency__}'
    status, output = getstatusoutput(cmd)
    if status == 0:
        return True
    raise Exception(f'Error occurs when executing {cmd}:\nStatus Code:{status}\nOutput:{output}')


def set_packet_loss_rate(network_interface: str, loss_limit: str):
    # network_interface master: eth0, worker: wlan0
    pass
    cmd = f'sudo tc qdisc add dev {network_interface} root netem loss {loss_limit}'  # 添加丢包规则


def delete_current_tc_qdisc(network_interface: str):
    pass
    # network_interface master: eth0, worker: wlan0
    cmd = f''
