import os
import time
import paramiko
from scp import SCPClient
from typing import List
from datetime import datetime
from tqdm import tqdm  # ✅ 正确导入
import json

def merge_and_transfer_lists(
        list_a: List[str], 
        list_b: List[str],
        target_ip: str = "192.168.2.78",
        remote_dir: str = "/nfs/ai/send_weight/send_lists",
        ssh_key_path: str = "~/.ssh/id_ed25519"
    ):
    """合并两个列表并传输到远程服务器"""
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"merged_list_{timestamp}.json"
    local_dir = "./send_weights_list"
    os.makedirs(local_dir, exist_ok=True)  # 关键修复1：确保目录存在
    local_path = os.path.join(local_dir, filename)
    
    # 合并数据并保存到本地
    try:
        merged_list = list_a + list_b
        with open(local_path, 'w') as f:
            json.dump(merged_list, f, indent=2)
    except Exception as e:
        print(f"❌ 本地文件保存失败: {str(e)}")
        return  # 关键修复2：保存失败时提前返回

    # 初始化SSH客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 建立SSH连接
        ssh.connect(
            target_ip, 
            port=22,
            username="root",
            key_filename=os.path.expanduser(ssh_key_path)
        )
        
        # 确保远程目录存在
        ssh.exec_command(f"mkdir -p {remote_dir}")
        
        # 传输文件带进度条
        file_size = os.path.getsize(local_path)
        with tqdm(total=file_size, 
              unit='B', 
              unit_scale=True, 
              desc=filename) as pbar:
            def _progress_callback(filename: str, sent: int, total: int):
                """SCP传输进度回调"""
                pbar.update(sent - pbar.n)
                pbar.set_postfix(file=os.path.basename(filename))  # 显示当前传输的文件名
                
            with SCPClient(ssh.get_transport(), progress=_progress_callback) as scp:
                scp.put(local_path, remote_dir)
        
        # 验证传输结果
        stdin, stdout, stderr = ssh.exec_command(f"md5sum {os.path.join(remote_dir, filename)}")
        remote_md5 = stdout.read().decode().split()[0]
        local_md5 = os.popen(f"md5sum {local_path}").read().split()[0]
        
        if remote_md5 == local_md5:
            print(f"\n✅ 传输验证成功 | 文件路径: {remote_dir}/{filename}")
        else:
            raise Exception("MD5校验失败，文件可能损坏")
            
    except Exception as e:
        print(f"\n❌ 传输失败: {str(e)}")
        # 清理残留文件
        ssh.exec_command(f"rm -f {os.path.join(remote_dir, filename)}")
    finally:
        ssh.close()
        os.remove(local_path)  # 清理本地临时文件
