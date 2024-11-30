import paramiko

remote_ip = '192.168.1.102'
username = 'pi'
password = '88888888'


def kill_remote_python_process():
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


kill_remote_python_process()
