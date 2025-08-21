import paramiko
from scp import SCPClient
import os
import time
from tqdm import tqdm
import json
from datetime import datetime, timedelta

def check_remote_path_exists(
    ip: str, 
    remote_path: str, 
    key_path: str, 
    path_type: str = "any"
) -> bool:
    """
    检查远程路径是否存在
    :param ip: 目标服务器IP
    :param remote_path: 要检查的远程路径
    :param key_path: SSH私钥路径
    :param path_type: 检测类型，可选值: 
        "any" - 存在即可（默认）
        "dir" - 必须是目录
        "file" - 必须是文件
    :return: 是否存在
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ip, 22, "root", key_filename=key_path)
        
        # 根据检测类型生成命令
        if path_type == "dir":
            test_cmd = f"test -d {remote_path}"
        elif path_type == "file":
            test_cmd = f"test -f {remote_path}"
        else:
            test_cmd = f"test -e {remote_path}"
        
        stdin, stdout, stderr = ssh.exec_command(f"{test_cmd} && echo 'EXISTS'")
        return 'EXISTS' in stdout.read().decode().strip()
    finally:
        ssh.close()



# 缓存文件路径（可配置）
CACHE_FILE = "ssh_query_cache.json"

def load_cache():
    """加载缓存数据"""
    if not os.path.exists(CACHE_FILE):
        return {}
    
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"缓存文件损坏，已重置: {str(e)}")
        return {}

def save_cache(cache):
    """保存缓存数据"""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_cache_key(ip, path, path_type):
    """生成唯一缓存键"""
    return f"{ip}|{path}|{path_type}"

def cached_check(ip, path, path_type, key_path):
    """带缓存的路径检查"""
    # 加载缓存
    cache = load_cache()
    
    # 生成缓存键
    cache_key = get_cache_key(ip, path, path_type)
    
    # 检查缓存有效性
    if cache_key in cache:
        cached_time = datetime.fromisoformat(cache[cache_key]["timestamp"])
        if datetime.now() - cached_time < timedelta(minutes=2):
            print(f"使用缓存结果 [{ip}] {path}")
            return cache[cache_key]["result"]
    
    # 执行实际检查
    start_time = time.time()
    result = check_remote_path_exists(ip, path, key_path, path_type)
    query_time = time.time() - start_time
    
    # 更新缓存
    cache[cache_key] = {
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "query_time": query_time,
        "path_type": path_type
    }
    save_cache(cache)
    
    return result

def check_remote_dir_exists(ip, remote_path, key_path):
    return cached_check(ip, remote_path, "dir", key_path)

def check_remote_file_exists(ip, remote_path, key_path):
    return cached_check(ip, remote_path, "file", key_path)

