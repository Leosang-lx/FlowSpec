import paramiko

remote_ip = '192.168.1.164'
username = 'nvidia'
password = 'nvidia'

# multiple_ip = [
#     '192.168.1.161',
#     '192.168.1.162',
#     '192.168.1.163',
#     '192.168.1.164'
# ]

multiple_ip = [
    '100.107.85.75',
    '100.126.236.128',
    '100.66.107.18',
    '100.98.149.22'
]

def kill_remote_python_process(remote_ip):
    ssh = paramiko.SSHClient()

    # add to known_hosts
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # connect ssh
    ssh.connect(remote_ip, port=22, username=username, password=password)

    stdin, stdout, stderr = ssh.exec_command("ps -a | grep python | grep -v grep | awk '{print $1}' | xargs kill -9")
    output = stdout.read().decode()
    error = stderr.read().decode()

    print("Output: ", output)
    if error:
        print("Error")

    ssh.close()


def kill_multiple(multiple_ip):
    for ri in multiple_ip:
        kill_remote_python_process(ri)

# kill_remote_python_process(remote_ip)
kill_multiple(multiple_ip)
