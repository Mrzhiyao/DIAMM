import paramiko
from scp import SCPClient
import os
import time
from tqdm import tqdm
from query_pods import query_pods_info


class ProgressTracker:
    def __init__(self, total_size):
        self.pbar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}] {remaining}",
            desc="总进度"
        )
        self.start_time = time.time()
        self.transferred = 0
        self.current_file = None

    def callback(self, filename, size, sent):
        if self.current_file != filename:
            self.current_file = filename
            tqdm.write(f"\n📁 正在传输文件: {os.path.basename(filename)}")
        
        chunk = sent - self.transferred
        self.transferred += chunk
        self.pbar.update(chunk)
        
        # 实时速度计算
        elapsed = time.time() - self.start_time
        speed = self.transferred / elapsed / 1024 / 1024 if elapsed > 0 else 0
        self.pbar.set_postfix(speed=f"{speed:.2f} MB/s")

def check_remote_dir_exists(ip: str, remote_path: str, key_path: str) -> bool:
    """检查远程路径是否存在（存在则跳过）"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ip, 22, "root", key_filename=key_path)
        stdin, stdout, stderr = ssh.exec_command(f"test -d {remote_path} && echo 'EXISTS'")
        return 'EXISTS' in stdout.read().decode().strip()
    finally:
        ssh.close()

def scp_to_servers(ips: list, local_path: str, remote_base_path: str):
    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
    folder_name = os.path.basename(local_path.rstrip('/'))
    remote_full_path = os.path.join(remote_base_path, folder_name)
    flag_name = f"{folder_name}_transfer_complete.flag"
    MIN_SPEED = 80 * 1024 * 1024  # 100 MB/s in bytes/s
    CHECK_INTERVAL = 2  # 每5秒检查一次速度

    # 预计算总大小
    total_size = 0
    for root, _, files in os.walk(local_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))

    for ip in ips:
        if check_remote_dir_exists(ip, remote_full_path, ssh_key_path):
            print(f"⏩ [{ip}] 远程路径 {remote_full_path} 已存在，跳过传输")
            continue

        tracker = ProgressTracker(total_size)
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, 22, "root", key_filename=ssh_key_path)
            
            # 创建远程目录（确保基础目录存在）
            ssh.exec_command(f"mkdir -p {remote_base_path} {remote_full_path}")
            
            # 传输文件夹内容
            with SCPClient(ssh.get_transport(), progress=tracker.callback) as scp:
                for item in os.listdir(local_path):
                    local_item = os.path.join(local_path, item)
                    scp.put(local_item, remote_full_path, recursive=True)
            
            # ================ 新增权限设置步骤 ================
            # 递归修改目录权限
            print(f"🛠️  [{ip}] 正在设置目录权限 (chmod -R 777)")
            stdin, stdout, stderr = ssh.exec_command(
                f"chmod -R 777 {remote_full_path}",
                get_pty=True  # 需要伪终端来执行权限操作
            )
            
            # 检查命令执行结果
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                raise Exception(f"权限设置失败: {error_msg}")

            # ----------------- 关键修改：标志文件写入基础目录 -----------------
            complete_flag_content = f"""Transfer Complete
            Folder: {folder_name}
            Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
            TotalSize: {total_size} bytes"""
            
            # 确保标志文件路径为 /home/yaozhi/images/xxx.flag
            flag_path = os.path.join(remote_base_path, flag_name)  # 基础目录下
            ssh.exec_command(f"echo '{complete_flag_content}' > {flag_path}")
            # ----------------------------------------------------------
            
            tracker.pbar.close()
            print(f"\n✅ [{ip}] 传输完成 | 模型路径: {remote_full_path} | 标志文件: {flag_path}")
        except Exception as e:
            tracker.pbar.close()
            print(f"\n❌ [{ip}] 传输失败: {str(e)}")
            try:
                ssh.exec_command(f"rm -rf {remote_full_path} {flag_path}")
                print(f"已清理残留文件: {remote_full_path} 和 {flag_path}")
            except:
                pass
        finally:
            ssh.close()
    

if __name__ == "__main__":
    target_ips = ["192.168.2.80"]
    local_folder = "/nfs/ai/ai-model/Qwen2.5-3B-ollama"
    remote_base_dir = "/home/yaozhi/images"
    scp_to_servers(target_ips, local_folder, remote_base_dir)
