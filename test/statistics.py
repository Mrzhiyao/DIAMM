import os
import time
import subprocess
from datetime import datetime, timedelta

def get_folder_size(folder_path):
    """计算文件夹总大小（单位：MB）"""
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                continue
    return round(total_size / (1024 * 1024), 2)

def get_gpu_info():
    """
    获取GPU信息（显存使用量 + GPU利用率）
    返回: (显存使用量(MB), GPU利用率(%))
    """
    try:
        # 同时查询显存和GPU利用率
        output = subprocess.check_output(
            ["nvidia-smi", 
             "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,nounits,noheader"],
            encoding="utf-8"
        ).strip()
        
        # 解析双指标数据（示例输出："123, 45"）
        mem_used, gpu_util = output.split(',')
        return int(mem_used), int(gpu_util.strip())
    
    except Exception as e:
        return f"Error: {str(e)}", "N/A"

def clean_old_data(log_file, hours=12):
    """清理指定小时前的旧数据（兼容新旧日志格式）"""
    if not os.path.exists(log_file):
        return
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    temp_file = log_file + ".tmp"
    
    try:
        with open(log_file, 'r') as fin, open(temp_file, 'w') as fout:
            header = fin.readline().strip()
            
            # 处理可能存在的旧版日志格式
            if "gpu_utilization" not in header:
                header = "timestamp,folder_size(MB),gpu_memory(MB),gpu_utilization(%)"
            
            fout.write(header + "\n")
            
            for line in fin:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                
                timestamp_str = parts[0]
                try:
                    record_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    if record_time > cutoff_time:
                        fout.write(line)
                except ValueError:
                    continue
        
        os.replace(temp_file, log_file)
    except Exception as e:
        print(f"清理旧数据失败: {str(e)}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def monitor_resources(folder_path, log_file="monitor_log.csv", interval=1):
    """监控资源使用情况"""
    # 创建包含GPU利用率的新版日志头
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,folder_size(MB),gpu_memory(MB),gpu_utilization(%)\n")
    
    try:
        while True:
            clean_old_data(log_file)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            folder_size = get_folder_size(folder_path)
            gpu_data = get_gpu_info()
            
            # 统一处理GPU数据（兼容有无GPU的情况）
            if isinstance(gpu_data, tuple):
                gpu_mem, gpu_util = gpu_data
            else:
                gpu_mem, gpu_util = gpu_data, "N/A"
            
            with open(log_file, "a") as f:
                f.write(f"{timestamp},{folder_size},{gpu_mem},{gpu_util}\n")
            
            # 优化控制台输出格式
            console_msg = f"[{timestamp}] Folder: {folder_size:>7}MB | "
            console_msg += f"GPU Mem: {str(gpu_mem):>5}MB | "
            console_msg += f"Utilization: {str(gpu_util):>3}%"
            print(console_msg)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    target_folder = "/home/images"
    
    if not os.path.isdir(target_folder):
        raise ValueError(f"无效的文件夹路径: {target_folder}")
    
    monitor_resources(
        folder_path=target_folder,
        log_file="resource_monitor.csv",
        interval=1
    )
