import paramiko
import os

# 目标服务器信息
hostname = "192.168.2.190"  # 修改为目标 IP
user = "root"
key = os.path.expanduser("~/.ssh/id_ed25519")
folder_path = "/home/yaozhi/images/yz_diffusion"

# 数据结构示例
data = {
    '192.168.2.190': {
        'text2image': {'/home/yaozhi/images/yz_diffusion': 1},
        'image2image': {'/home/yaozhi/images/yz_diffusion': 1},
        'text/image2image': {'/home/yaozhi/images/yz_diffusion': 1}
    }
}


def delete_remote_folder(host, path):
    """通过 SSH 删除远程文件夹"""
    try:
        # 创建 SSH 客户端
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 加载私钥（如果密钥有密码需要添加 password 参数）
        private_key = paramiko.Ed25519Key.from_private_key_file(key)
        
        # 连接服务器
        client.connect(hostname=host, username=user, pkey=private_key)
        
        clean_path = path.rstrip('/')
        flag_path = f"{clean_path}_transfer_complete.flag"

        # 执行删除命令
        command = f"rm -rf {clean_path} && rm -f {flag_path}"
        stdin, stdout, stderr = client.exec_command(command)
        
        # 检查错误
        if stderr.read():
            print(f"删除失败: {stderr.read().decode()}")
        else:
            print(f"成功删除 {path}")
            
        client.close()
    except Exception as e:
        print(f"连接或操作失败: {str(e)}")

