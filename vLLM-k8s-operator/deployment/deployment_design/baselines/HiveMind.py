from sql import get_slack_time, get_t_queue
from generate_task import generate_task_queue
import copy
from collections import Counter
from query_pods import query_pods_info
# from generate_queue import generate_unique_ratios
from asplos24_generate_task import generate_task_batch
from asplos24_read_tasks import read_tasks
import time
from test_main import available_ip
from datetime import datetime
import random
from uxcost import UXCostOptimizer
from collections import defaultdict
import os 
import json
import re 
import argparse
from customize_yaml_text2text import modify_text2text_yaml_template, local_modify_text2text_yaml_template
from customize_yaml_text2text_ollama import modify_text2text_ollama_yaml_template, local_modify_text2text_ollama_yaml_template
from customize_yaml_image2text import modify_image2text_yaml_template, local_modify_image2text_yaml_template
from customize_yaml_text2image import modify_text2image_yaml_template, local_modify_text2image_yaml_template
from customize_yaml_text2video import modify_text2video_yaml_template, local_modify_text2video_yaml_template
import subprocess
import shlex
import redis
from check_node_weight import check_remote_dir_exists, check_remote_file_exists
import math
from check_celery import get_non_success_tasks, get_container_online_counts, get_container_online_new
from query_gpu import query_prometheus_gpu
from send_weight_list import merge_and_transfer_lists
from bingfa import send_tasks
from read_all_json import process_json_files
from chech_filtered_subitems_error import filter_duplicate_pods, remove_duplicate_tasks


# 复用你已有的连接池配置
REDIS_POOL = redis.ConnectionPool(
    host='192.168.2.75',
    port=6379,
    db=1,
    password='123456',
    max_connections=80,
    decode_responses=False  # 保持二进制格式
)

weight_text2text_params = [6000+1000,14000+1000,16000+1000,18000+1000]
weight_image2text_params = [6000,16000]

weight_text2image_params = [4000]
weight_image2image_params = [4000]
weight_text2video_params = [19000]

param_path = {
    'text2text':{weight_text2text_params[0]:'/nfs/ai/ai-model/Qwen2.5-3B-Instruct',
                    weight_text2text_params[1]:'/nfs/ai/ai-model/Qwen2.5-7B-Instruct',
                    weight_text2text_params[2]:'/nfs/ai/ai-model/Meta-Llama-3.1-8B-Instruct',
                    weight_text2text_params[3]:'/nfs/ai/ai-model/glm-4-9b-chat',
                    weight_text2text_params[0]/2:'/nfs/ai/ai-model/Qwen2.5-3B-ollama',
                    weight_text2text_params[1]/2:'/nfs/ai/ai-model/Qwen2.5-7B-ollama',
                    weight_text2text_params[2]/2:'/nfs/ai/ai-model/llama31-8b-ollama',
                    weight_text2text_params[3]/2:'/nfs/ai/ai-model/glm4-9b-ollama'
                    },
    'image2text':{weight_image2text_params[0]:'/nfs/ai/ai-model/Qwen2-VL-2B-Instruct', 
                    weight_image2text_params[1]:'/nfs/ai/ai-model/Qwen2-VL-7B-Instruct'
                    },
    'text2image':{weight_text2image_params[0]:'/nfs/ai/ai-model/yz_diffusion'},
    'image2image':{weight_text2image_params[0]:'/nfs/ai/ai-model/yz_diffusion'},
    'text/image2image':{weight_text2image_params[0]:'/nfs/ai/ai-model/yz_diffusion'},
    'text2video':{weight_text2video_params[0]: '/nfs/ai/ai-model/CogVideo-1.0'}
    }

def sync_save_tasks(tasks_data: dict, expire_seconds: int = 300):
    """
    同步插入数据到 Redis 并设置过期时间
    :param tasks_data: 嵌套字典结构任务数据
    :param expire_seconds: 过期时间（秒），默认 600 秒=10 分钟
    """
    # print('tasks_data', tasks_data)
    # 从连接池获取连接
    r = redis.Redis(connection_pool=REDIS_POOL)
    try:
        # for task_type, task_info in tasks_data.items():
        #     print(task_type, task_info)

        for task_type, task_info in tasks_data.items():
            for task in task_info:
                # print('task_info', task)
                task_id = task.get("task_id")
                if not task_id:
                    print(f"警告: {task_type} 任务缺少 task_id，已跳过")
                    continue

                # 序列化为 JSON 二进制数据
                try:
                    serialized_data = json.dumps(task).encode('utf-8')
                except TypeError as e:
                    print(f"序列化失败（{task_id}）: {str(e)}")
                    continue

                # 插入数据并设置过期
                try:
                    r.set(
                        name=f"task:{task_id}",  # Key 格式: task:4045d042...
                        value=serialized_data,
                        ex=expire_seconds  # 设置过期时间
                    )
                    # print(f"已同步插入 task_id={task_id}")
                except redis.RedisError as e:
                    print(f"Redis 操作失败（{task_id}）: {str(e)}")
    finally:
        r.close()  # 将连接归还连接池



# 模型配置（保持不变）
MODEL_CONFIG = {
    "qwen-3b": {
        "gpu_mem": 6,   # GB
        "ram": 15,
        "task_types": ["text2text"],
        "implement_cost": 5,
        "max_task_process": 40,  # 高并发模型
        "concurrency_type": "high",  # 高并发
        "delay_factor": 0.1  # 超出后的延迟系数
    },
    "qwen-vl-2b": {
        "gpu_mem": 4,  
        "ram": 10,
        "task_types": ["image2text"],
        "implement_cost": 20,
        "max_task_process": 20,  # 高并发模型
        "concurrency_type": "high",
        "delay_factor": 0.15
    },
    "stable-diffusion": {
        "gpu_mem": 4,
        "ram": 6,
        "task_types": ["text2image", "image2image"],
        "implement_cost": 15,
        "concurrency_type": "queue",  # 排队模型
        "queue_factor": 0.05  # 排队延迟系数
    },
    "cogvideo-2b": {
        "gpu_mem": 18,
        "ram": 10,
        "task_types": ["text2video"],
        "implement_cost": 8,
        "concurrency_type": "serial"  # 串行模型
    }
}

NODES_INI = {
    '192.168.2.75': {'Name': 'b410-4090d-1', 'GPU': 24500}, 
    '192.168.2.78': {'Name': 'b410-4090d-3', 'GPU': 24500}, 
    '192.168.2.80': {'Name': 'b410-3090-1', 'GPU': 24500},
    '192.168.2.5': {'Name': 'b410-2070s-1', 'GPU': 8000},
    '192.168.2.6': {'Name': 'b410-2070s-2', 'GPU': 8000},
    '192.168.2.7': {'Name': 'b410-2070s-3', 'GPU': 8000}
}

# 集群节点信息（新增）
NODES = {
    '192.168.2.75': {
        'Name': 'b410-4090d-1', 
        'Memory': 64, 
        'GPU': 24000, 
        'Memory_Used': 20.33, 
        'Memory_Available': 43.67, 
        'GPU_Available': '16144'}, 
    '192.168.2.190': {
        'Name': 'b410-4090d-2', 
        'Memory': 64, 
        'GPU': 24000, 
        'Memory_Used': 13.85, 
        'Memory_Available': 50.15, 
        'GPU_Available': '7167'}, 
    '192.168.2.78': {
        'Name': 'b410-4090d-3', 
        'Memory': 64, 
        'GPU': 24000, 
        'Memory_Used': 12.49, 
        'Memory_Available': 51.51, 
        'GPU_Available': '24072'}, 
    '192.168.2.80': {
        'Name': 'b410-3090-1', 
        'Memory': 32, 
        'GPU': 24000, 
        'Memory_Used': 7.11, 
        'Memory_Available': 24.89, 
        'GPU_Available': '18555'},

    '192.168.2.5': {
        'Name': 'b410-2070s-1', 
        'Memory': 32, 
        'GPU': 8000, 
        'Memory_Used': 3.84, 
        'Memory_Available': 60.16, 
        'GPU_Available': '7883'},
    '192.168.2.6': {
        'Name': 'b410-2070s-2', 
        'Memory': 32, 
        'GPU': 8000, 
        'Memory_Used': 3.84, 
        'Memory_Available': 60.16, 
        'GPU_Available': '7883'},
    '192.168.2.7': {
        'Name': 'b410-2070s-3', 
        'Memory': 32, 
        'GPU': 8000, 
        'Memory_Used': 3.84, 
        'Memory_Available': 60.16, 
        'GPU_Available': '7883'}
}

NODES_IP = {
    'b410-4090d-1': '192.168.2.75',
    'b410-4090d-2': '192.168.2.190',
    'b410-4090d-3': '192.168.2.78',
    'b410-3090-1': '192.168.2.80',
    'b410-2070s-1': '192.168.2.5',
    'b410-2070s-2': '192.168.2.6',
    'b410-2070s-3': '192.168.2.7',
    # 'b410-2070s-4': '192.168.2.133',
}


local_weight_paths = {
    'text2text':{
                '/home/yaozhi/images/glm-4-9b-chat',
                '/home/yaozhi/images/glm4-9b-ollama',
                '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct',
                '/home/yaozhi/images/llama31-8b-ollama',
                '/home/yaozhi/images/Qwen2.5-7B-Instruct',
                '/home/yaozhi/images/Qwen2.5-7B-ollama',
                '/home/yaozhi/images/Qwen2.5-3B-Instruct',
                '/home/yaozhi/images/Qwen2.5-3B-ollama',
                },
    'image2text':{'/home/yaozhi/images/Qwen2-VL-7B-Instruct', 
                '/home/yaozhi/images/Qwen2-VL-2B-Instruct'
                },
    'text2image':{'/home/yaozhi/images/yz_diffusion'},
    'image2image':{'/home/yaozhi/images/yz_diffusion'},
    'text/image2image':{'/home/yaozhi/images/yz_diffusion'},
    'text2video':{'/home/yaozhi/images/CogVideo-1.0'}
}

ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
remote_full_path = '/home/yaozhi/images/'
flag = 'transfer_complete.flag'

task_type  = 'text2text'

inference_framework = {
    'text2text': {'Turbomind', 'ollama'}, 
    'image2text': {'Turbomind'}, 
    'text2image': {'Xformers'}, 
    'image2image': {'Xformers'}, 
    'text2video': {'SwissArmyTransformer'}, 
}

# 因为后面计算已部署模型会减这部分，所以提前加权
# 1 1 5 10 20 40 60 80
text2text_delay = [10.08, 5.07, 5.07, 5.08, 5.07, 5.09, 5.14, 5.33]
text2text_delay = [x + 5 for x in text2text_delay]

image2text_delay = [15.06, 0.92, 5.09, 5.11, 5.13, 7.31, 12.45, 14.28]
image2text_delay = [x + 20 for x in image2text_delay]

text2image_delay = [23.28, 2.23, 5.06, 8.09, 13.35, 19.66, 19.20, 18.97]
text2image_delay = [x + 15 for x in text2image_delay]

image2image_delay = [20.15, 1.20, 5.08, 7.63, 11.43, 17.84, 23.13, 33.70]
image2image_delay = [x + 15 for x in image2image_delay]

text2video_delay = [45, 45, 45*2, 45*3, 45*4, 45*5, 45*6, 45*7]
text2video_delay = [x + 8 for x in text2video_delay]

text2text_delay_ollama = [10.14,5.19,5.14,5.15,5.27,7.4,9.21,10.41]
text2text_delay_ollama = [x + 5 for x in text2text_delay_ollama]

text2text_params = [3,7,8,9]
image2text_params = [2,7]
text2image_params = [3]
image2image_params = [3]
text2video_params = [18]

double_text2text_params = [20000+1000, 20000+500, 18000+2000, 8000+1000]
double_image2text_params = [16000, 6000]
double_text2image_params = [4000]
double_image2image_params = [4000]
double_text2video_params = [19000]

max_text2text_nums = 100
max_image2text_nums = 60
# max_text2image_nums = 40
# max_image2image_nums = 40
# max_text2image_nums = 20
# max_image2image_nums = 20
max_text2image_nums = 10
max_image2image_nums = 10


max_text2video_nums = 1
# max_total = 100 + 60 + (40 + 40)*2 
max_total = 100 + 60 + (20 + 20)*2 
# 一个任务在不同部署方式，所有并行压力（最优和最差情况）和处理途径下的平均完成时间
# 默认4090 3090相同，2070的推理能力比：2:1
# Turbomind和ollama的推理延迟偏好也是2:1

def process_all_tasks(tt_data):
    # 所有需要处理的任务类型
    task_types = ['text2text', 'image2text', 'text2image', 'image2image', 'text2video']
    
    # 合并所有任务类型的数据
    all_tasks = []
    for task_type in task_types:
        if task_type in tt_data:
            all_tasks.extend(tt_data[task_type])
    
    # 按照wait_time从大到小排序
    sorted_tasks = sorted(
        all_tasks, 
        key=lambda x: x['wait_time'], 
        reverse=True
    )
    
    return sorted_tasks


# ASPLOS24


def deduplicate_and_count_tasks(tasks):
    seen = {}  # 结构：{task_type: {"row": int, "col": int, "count": int}}
    
    for task in tasks:
        task_type = task['task_type']
        if task_type not in seen:
            # 首次出现时记录位置并初始化计数器
            seen[task_type] = {
                "row": task['row'],
                "col": task['col'],
                "count": 1
            }
        else:
            # 后续出现只增加计数器
            seen[task_type]["count"] += 1
    
    # 转换数据结构到输出格式
    return [{
        "task_type": k,
        "row": v["row"],
        "col": v["col"],
        "count": v["count"]
    } for k, v in seen.items()]

def map_models(tasks):
    # 定义参数映射规则
    param_mapping = {
        'text2text': [3, 7, 8, 9],
        'image2text': [2, 7],
        'text2image': [3],
        'image2image': [3]
    }
    
    corr_numbers = [1,1,5,10,20,40,60,80]

    result = []
    
    for task in tasks:
        task_type = task['task_type']
        row = task['row']
        col = task['col']
        counts = task['count']
        
        # 根据不同类型选择模型参数
        if task_type in ['text2text', 'image2text']:
            # 确保row在参数列表范围内
            params = param_mapping[task_type]
            index = min(row, len(params)-1)  # 防止索引越界
            model = f"{params[index]}b"
        elif task_type == 'text2video':
            model = "cogvideox-2b-sat"
        else:
            model = "stable-diffusion"
        
        # 决策要部署多少模型
        if task_type == 'text2text':
            if col <8:
                frame = 'default'
                model_number = 1 + int(counts/corr_numbers[7])
            else:
                frame = 'ollama'
                model_number = 1 + int(counts/corr_numbers[7])
        
        elif task_type == 'image2text':
            frame = 'default'
            if col<6:
                model_number = 1 + int(counts/corr_numbers[6])
            else:
                model_number = 1
            print('ceshimodel_number_image2text', model_number)

        elif task_type == 'text2image' or task_type =='image2image': 
            frame = 'default'
            # print('col', col)
            if col<5:
                model_number = 1 + int(counts/corr_numbers[5])
            else:
                model_number = 1
            print('ceshimodel_number_text2image', model_number)

        elif task_type == 'text2video':
            frame = 'default'
            if counts>corr_numbers[0]:
                model_number = counts
            else:
                model_number = 1
        result.append({
            'models': task_type,
            'container': model,
            'model_number': model_number,
            'frame': frame,
            'task_counts': counts
        })
    
    return result

# 固定保存目录
SAVE_DIR = "./tasks_ip"  # 可根据需求修改路径

def save_classified_to_timestamped_file(classified_data):
    """保存数据到带时间戳的 JSON 文件"""
    try:
        # 确保目录存在
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # 生成带时间戳的文件名（格式：tasks_年月日_时分秒.json）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tasks_{timestamp}.json"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # 转换时间字段为字符串（如果包含 datetime 对象）
        for task_type in classified_data:
            for task in classified_data[task_type]:
                if isinstance(task.get("create_time"), datetime):
                    task["create_time"] = task["create_time"].isoformat()
        
        # 保存文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(classified_data, f, ensure_ascii=False, indent=2)
            
        print(f"数据已保存到：{filepath}")
        return filepath
    except Exception as e:
        print(f"保存失败：{str(e)}")
        return None

def load_latest_classified():
    """加载最新保存的数据"""
    try:
        # 获取所有 tasks_ 开头的 JSON 文件
        task_files = [
            f for f in os.listdir(SAVE_DIR) 
            if f.startswith("tasks_") and f.endswith(".json")
        ]
        if not task_files:
            print("目录中无历史数据")
            return None
        
        # 按文件名中的时间戳排序（最新在前）
        task_files.sort(reverse=True)
        latest_file = os.path.join(SAVE_DIR, task_files[0])
        
        # 读取文件
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 转换时间字段为 datetime 对象（可选）
        for task_type in data:
            for task in data[task_type]:
                if isinstance(task.get("create_time"), str):
                    task["create_time"] = datetime.fromisoformat(task["create_time"])
        
        # print(f"已加载最新文件：{latest_file}")
        return data
    except Exception as e:
        print(f"加载失败：{str(e)}")
        return None

def load_all_classified():
    """加载目录下全部数据"""
    all_data = {}
    try:
        for filename in os.listdir(SAVE_DIR):
            if filename.startswith("tasks_") and filename.endswith(".json"):
                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_data[filename] = data
        return all_data
    except Exception as e:
        print(f"加载失败：{str(e)}")
        return None


def calculate_non_pending_percentage(all_data):
    # 合并 text2image 和 image2image 任务类型
    merged_data = defaultdict(list)
    for filename, classified in all_data.items():
        for task_type, tasks in classified.items():
            merged_type = task_type
            if task_type in ["text2image", "image2image"]:
                merged_type = "text/image2image"
            merged_data[merged_type].extend(tasks)

    # 统计结构：task_type → (service_ip, container, frame) → {total, non_pending}
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "non_pending": 0}))
    
    for task_type, tasks in merged_data.items():
        for task in tasks:
            # 提取三个字段（假设字段存在，否则需处理 KeyError）
            service_ip = task["service_ip"]
            container = task["container"]  # 新增字段
            frame = task["frame"]           # 新增字段
            key = (service_ip, container, frame)
            
            stats[task_type][key]["total"] += 1
            if task["status"] != "Pending":
                stats[task_type][key]["non_pending"] += 1

    # 结果结构：task_type → {"service_ip": ..., "container": ..., "frame": ..., "percentage": ...}
    result = defaultdict(list)
    for task_type, keys in stats.items():
        for key in keys:
            service_ip, container, frame = key
            counts = stats[task_type][key]
            total = counts["total"]
            non_pending = counts["non_pending"]
            percentage = (non_pending / total) * 100 if total > 0 else 0.0
            
            result[task_type].append({
                "service_ip": service_ip,
                "container": container,
                "frame": frame,
                "non_pending_percentage": percentage,
                "total_tasks": total,
                "non_pending_tasks": non_pending
            })
    
    return dict(result)

def find_low_percentage_entries(data, input_percent):
    results = {}
    
    # 遍历每个分类（text2text, image2text 等）
    for category, entries in data.items():
        # 计算该分类的总任务数
        total = sum(entry['total_tasks'] for entry in entries)
        if total == 0:
            continue
            
        # 筛选符合 10%-20% 比例的条目
        filtered = []
        for entry in entries:
            percentage = (entry['total_tasks'] / total) * 100
            if percentage < input_percent:
                filtered.append({
                    **entry,
                    'percentage': round(percentage, 2)
                })
        
        if filtered:
            results[category] = {
                'total_tasks': total,
                'matched_entries': filtered
            }
            
    return results


def check_and_get_files(paths):
    """
    获取多个路径下所有文件的完整路径列表
    :param paths: 一个或多个文件夹路径
    :return: 所有文件的完整路径列表
    """
    all_files = []
    
    for path in paths:
        # 检查路径是否存在
        if not os.path.exists(path):
            print(f"警告: 路径 '{path}' 不存在，已跳过")
            continue
            
        # 检查是否为目录
        if not os.path.isdir(path):
            print(f"警告: '{path}' 不是目录，已跳过")
            continue
        
        # 遍历目录中的所有条目
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            
            # 只添加文件，忽略目录
            if os.path.isfile(entry_path):
                all_files.append(entry_path)
    
    return all_files




import shutil

def get_num(data):
    safe_result = []
    for item in data:
        container = item.get("container", "")
        if "-" in container:
            last_part = container.rsplit("-", 1)[-1]
            if last_part.isdigit():
                safe_result.append(int(last_part))
            else:
                safe_result.append(None)  # 最后一部分不是数字
        else:
            safe_result.append(None)  # 没有 "-"

    return safe_result

def generate_unique_number(exclude_list):
    """生成不包含在排除列表中的 0-100 随机数"""
    while True:
        num = random.randint(0, 99)
        if num not in exclude_list:
            return num

def select_and_update_machines(machines, query_value):
    # 筛选 Available_GPU 大于 query_value 的机器
    selected_machines = [m for m in machines if m['Available_GPU'] > query_value]
    return_machine = None
    # 打印筛选结果
    # print("符合条件的机器（更新前）:")
    # for machine in selected_machines:
    #     print(f"Hostname: {machine['Hostname']}, Available_GPU: {machine['Available_GPU']}")
    
    # 更新原列表中这些机器的 Available_GPU（减去 query_value）
    for machine in selected_machines:
        machine['Available_GPU'] -= query_value
        return_machine = machine
        break
    
    # 返回筛选出的机器列表（更新后的值）
    return selected_machines, return_machine


def sanitize_container_name(model_name: str, suffix: str) -> str:
    """生成符合 Kubernetes 规范的容器名称"""
    # Step 1: 转换全小写
    cleaned = model_name.lower()
    
    # Step 2: 替换非法字符为连字符
    cleaned = re.sub(r'[^a-z0-9-]', '-', cleaned)
    
    # Step 3: 合并连续连字符
    cleaned = re.sub(r'-+', '-', cleaned)
    
    # Step 4: 去除首尾连字符
    cleaned = cleaned.strip('-')
    
    # Step 5: 限制长度并拼接后缀
    max_base_length = 30  # 保留后缀空间
    final_name = f"{cleaned[:max_base_length]}-{suffix}"
    
    # 最终再次清理
    final_name = re.sub(r'[^a-z0-9-]', '', final_name)[:63]
    return final_name

def get_container_name(default, model, model_type, number):

    if model_type == 'text2image' or model_type == 'image2image':
        return "stable-diffusion"
    elif model_type == 'text2video':
        return sanitize_container_name("CogVideoX-2b-sat", number) 
    elif model_type == 'image2text':
        if '2b' in model:
            return sanitize_container_name("Qwen2-VL-2B-Instruct", number) 
        elif '7b' in model:
            return sanitize_container_name("Qwen2-VL-7B-Instruct", number) 

    elif model_type == 'text2text' and default == 'ollama':
        if '3b' in model or '3B' in  model:
            return sanitize_container_name("Qwen2.5-3B-ollama", number)
        elif '7b' in model or '7B' in model:
            return sanitize_container_name("Qwen2.5-7B-ollama", number)
        elif '8b' in model or '8B' in model:
            return sanitize_container_name("llama31-8b-ollama", number)
        elif '9b' in model or '9B' in model:
            return sanitize_container_name("glm4-9b-ollama", number)
    else:
        if '3b' in model or '3B' in model:
            return sanitize_container_name("Qwen2.5-3B-Instruct", number) 
        elif '7b' in model or '7B' in model:
            return sanitize_container_name("Qwen2.5-7B-Instruct", number) 
        elif '8b' in model or '8B' in model:
            return sanitize_container_name("Meta-Llama-3.1-8B-Instruct", number) 
        elif '9b' in model or '9B' in model:
            return sanitize_container_name("glm-4-9b-chat", number) 

def extract_unique_fields(data):
    unique_combinations = set()  # 存储所有唯一的组合
    result = []

    # 遍历所有可能的任务类型（如 text2text、image2text、text2image 等）
    for task_type in data:
        if isinstance(data[task_type], list):  # 确保该键对应的值是任务列表
            for entry in data[task_type]:
                # 提取目标字段
                key = (
                    entry.get("container", "N/A"),
                    entry.get("container_num", "N/A"),
                    entry.get("frame", "N/A"),
                    entry.get("service_ip", "N/A")
                )
                if key not in unique_combinations:
                    unique_combinations.add(key)
                    result.append({
                        "task_type": task_type,  # 记录任务来源类型
                        "container": key[0],
                        "container_num": key[1],
                        "frame": key[2],
                        "service_ip": key[3]
                    })
    return result

def develop_task(online_tasks, file_stats, tasks, classified, celery_task):

    service_develpoment = query_pods_info()
    filtered_query = [item for item in service_develpoment if item['pod'] != '']
    # print('filtered_query', filtered_query)
    # 查看容器部署时哪些数字可以用
    contain_num_text2text = [
        item for item in filtered_query
    if "text2text" in item.get("pod", "").lower()
    ]
    contain_num_image2text = [
        item for item in filtered_query
    if "image2text" in item.get("pod", "").lower()
    ]
    contain_num_text2image = [
        item for item in filtered_query
    if "text2image" in item.get("pod", "").lower()
    ]
    contain_num_image2image = [
        item for item in filtered_query
    if "image2image" in item.get("pod", "").lower()
    ]
    contain_num_text2video = [
        item for item in filtered_query
    if "text2video" in item.get("pod", "").lower()
    ]

    con_num_text2text = get_num(contain_num_text2text)
    con_num_image2text = get_num(contain_num_image2text)

    con_num_text2image = get_num(contain_num_text2image)
    con_num_image2image = get_num(contain_num_image2image)
    con_num_ti2image = con_num_text2image + con_num_image2image
    con_num_text2video = get_num(contain_num_text2video)
    

    # filepath = os.path.join(directory, filename)
    
    # 基于任务数量的模型部署决策（简化版，参考论文方法）
    # 查询已有模型的运行任务数
    if celery_task != None:
        used_task_number = get_container_online_counts(celery_task)
        print('used_task_number', used_task_number)
    else:
        used_task_number = {}
    
    # 获取各种任务类型的已使用数量
    used_task_number_text2text = used_task_number.get('text2text', 0)
    used_task_number_image2text = used_task_number.get('image2text', 0)
    used_task_number_text2image = used_task_number.get('text2image', 0)
    used_task_number_image2image = used_task_number.get('image2image', 0)
    used_task_number_text2video = used_task_number.get('text2video', 0)
    
    # 计算每种任务类型需要的模型实例数（基于论文方法）
    development_model = []
    maybe_machine = available_ip(float(3000))
    print('maybe_machine', maybe_machine)
    
    # text2video: 直接选择参数
    if file_stats['text2video'] != 0:
        for params_index in range(len(double_text2video_params)):
            selected_machines, machine = select_and_update_machines(maybe_machine, double_text2video_params[params_index])
            if machine != None:
                maybe_machine = selected_machines
                development_model.append(['text2video', double_text2video_params[params_index]])
                break
    
    # image2text: 基于任务数量选择参数（简化版）
    if file_stats['image2text'] != 0:
        # 简化：任务数少用小参数，任务数多用大参数
        if file_stats['image2text'] <= max_image2text_nums * 0.5:
            # 优先使用已有模型，否则选择小参数
            if len(contain_num_image2text) != 0:
                development_model.append(['image2text', double_image2text_params[1]])  # 小参数
            else:
                # 尝试部署小参数
                selected_machines, machine = select_and_update_machines(maybe_machine, double_image2text_params[1])
                if machine != None:
                    maybe_machine = selected_machines
                    development_model.append(['image2text', double_image2text_params[1]])
                else:
                    development_model.append(['image2text', double_image2text_params[0]])  # 大参数
        elif file_stats['image2text'] > max_image2text_nums:
            # 任务数较多，优先选择大参数
            selected_machines, machine = select_and_update_machines(maybe_machine, double_image2text_params[0])
            if machine != None:
                maybe_machine = selected_machines
                development_model.append(['image2text', double_image2text_params[0]])
            else:
                development_model.append(['image2text', double_image2text_params[1]])  # 资源不够用小参数
        else:
            # 中间范围：优先使用已有模型，否则尝试部署
            if len(contain_num_image2text) != 0:
                development_model.append(['image2text', double_image2text_params[1]])  # 使用已有模型，默认小参数
            else:
                # 尝试部署
                for params_index in range(len(double_image2text_params)):
                    selected_machines, machine = select_and_update_machines(maybe_machine, double_image2text_params[params_index])
                    if machine != None:
                        maybe_machine = selected_machines
                        development_model.append(['image2text', double_image2text_params[params_index]])
                        break

    # text2text: 基于任务数量选择参数（简化版）
    if file_stats['text2text'] != 0:
        if file_stats['text2text'] <= max_text2text_nums * 0.5:
            # 任务数少，优先小参数或已有模型
            if len(contain_num_text2text) != 0:
                # 使用已有模型
                development_model.append(['text2text', double_text2text_params[-1]])  # 最小参数
            else:
                # 尝试部署小参数
                for params_index in range(len(double_text2text_params)-1, -1, -1):  # 从后往前，小参数优先
                    selected_machines, machine = select_and_update_machines(maybe_machine, double_text2text_params[params_index])
                    if machine != None:
                        maybe_machine = selected_machines
                        development_model.append(['text2text', double_text2text_params[params_index]])
                        break
        else:
            # 任务数多，优先大参数
            for params_index in range(len(double_text2text_params)):
                selected_machines, machine = select_and_update_machines(maybe_machine, double_text2text_params[params_index])
                if machine != None:
                    maybe_machine = selected_machines
                    development_model.append(['text2text', double_text2text_params[params_index]])
                    break

    # text2image
    if file_stats['text2image'] != 0 or file_stats['image2image'] != 0:
        for params_index in range(len(double_text2image_params)):
            selected_machines, machine = select_and_update_machines(maybe_machine, double_text2image_params[params_index])
            if machine != None:
                # 更新服务器可用gpu信息
                maybe_machine = selected_machines
                development_model.append(['text2image', double_text2image_params[params_index]])
                break
    # 基于任务数量直接计算需要部署的模型数和分配（参考论文方法）
    # 不再使用复杂的评分矩阵，直接基于任务数量计算
    tasks_develop = []
    
    # 为每个任务类型计算需要部署的模型数并分配任务
    for task_type in ['text2text', 'image2text', 'text2image', 'image2image', 'text2video']:
        task_count = file_stats.get(task_type, 0)
        if task_count == 0:
            continue
        
        # 获取该任务类型的最大容量
        max_capacity_map = {
            'text2text': max_text2text_nums,
            'image2text': max_image2text_nums,
            'text2image': max_text2image_nums,
            'image2image': max_image2image_nums,
            'text2video': max_text2video_nums
        }
        max_capacity = max_capacity_map.get(task_type, 60)
        
        # 计算已有模型的剩余容量（简化：假设每个模型还有一半容量可用）
        existing_containers = []
        if task_type == 'text2text':
            existing_containers = contain_num_text2text
        elif task_type == 'image2text':
            existing_containers = contain_num_image2text
        elif task_type == 'text2image':
            existing_containers = contain_num_text2image
        elif task_type == 'image2image':
            existing_containers = contain_num_image2image
        elif task_type == 'text2video':
            existing_containers = contain_num_text2video
        
        # 计算已有模型还能处理的任务数（简化估算）
        existing_capacity = len(existing_containers) * max_capacity * 0.5
        
        # 计算需要的新模型数（基于论文方法）
        remaining_tasks = max(0, task_count - existing_capacity)
        new_models_needed = math.ceil(remaining_tasks / max_capacity) if remaining_tasks > 0 else 0
        
        # 为每个任务分配模型索引（简化：按顺序分配）
        for idx, task in enumerate(classified.get(task_type, [])):
            # 计算任务应该分配到哪个模型实例
            total_models = len(existing_containers) + new_models_needed
            if total_models > 0:
                model_idx = min(idx // max_capacity, total_models - 1)
            else:
                model_idx = 0
            tasks_develop.append({
                'task_type': task_type,
                'row': 0,  # 简化：不再需要行索引
                'col': model_idx  # 模型索引
            })

    print('tasks_develop (simplified)', len(tasks_develop))
    # 统计模型部署决策
    dedup_tasks = deduplicate_and_count_tasks(tasks_develop)
    time.sleep(1)
    # 模型部署方案：
    filtered = map_models(dedup_tasks)
    print('check_development', filtered)

    may_drop_models = []
    wait_tasks = []
    plan_tasks = []
    needs_gpu = []
    
    # 获取容器级别的任务计数（用于后续任务分配）
    if celery_task != None:
        used_container_task_number = get_container_online_new(celery_task)
        print('used_container_task_number', used_container_task_number)
    else:
        used_container_task_number = {}

    # stable模型因为共享模型，所以提前计数
    model_diffusion_number = 0
    model_diffusion_number_m = 0
    model_diffusion_number_t = 0
    for sub_model in filtered:
        model_name = sub_model['models']
        if model_name =='text2image' or model_name =='image2image':
            model_diffusion_number_m = model_diffusion_number_m + sub_model['model_number']

    for task_type, items in classified.items():
        if  task_type =='text2image' or task_type =='image2image':
            model_diffusion_number_t = model_diffusion_number_t + len(items)

    print('len(model_diffusion_number_t)', model_diffusion_number_t)
    # model_diffusion_number = min(model_diffusion_number_m, int(model_diffusion_number_t/max_text2image_nums)+1)
    model_diffusion_number = int((model_diffusion_number_t + used_task_number_text2image 
                                  + used_task_number_image2image)/max_text2image_nums)+1
 
    develop_text2image_items = [
        item for item in filtered
    if "text2image"  in item.get("models", "").lower()
    ]
    develop_image2image_items = [
        item for item in filtered
    if "image2image"  in item.get("models", "").lower()
    ]

    print('develop_text2image_items', develop_text2image_items)
    if len(develop_text2image_items) != 0 and len(develop_image2image_items) != 0:
        gen_img_tasks_number = develop_text2image_items[0]['task_counts'] + develop_image2image_items[0]['task_counts'] 
        acutal_task_counts = gen_img_tasks_number
    elif len(develop_text2image_items) != 0 and len(develop_image2image_items) == 0:
        gen_img_tasks_number = develop_text2image_items[0]['task_counts'] 
        acutal_task_counts = gen_img_tasks_number
    elif len(develop_text2image_items) == 0 and len(develop_image2image_items) != 0:
        gen_img_tasks_number = develop_image2image_items[0]['task_counts']    
        acutal_task_counts = gen_img_tasks_number
    else:
        print('sss')

    # 如果只需要一个stable_diffusion
    if model_diffusion_number == 1 and model_diffusion_number_t != 0:
        filtered_subitems = [
            item for item in filtered_query
        if 'text2image' in item.get("pod", "") and 'stable' in item.get("container", "").lower()
        ]
        filtered_subitems = filter_duplicate_pods(filtered_subitems)

        if len(filtered_subitems) == 0:
            # 当前集群无法提供该服务，需要部署
            # 在集群找能够满足该服务的服务器，优先推理速度更快的
            available_ips = available_ip(float(3)*1024 + 1000)

            sorted_available_ips = sorted(available_ips, key=lambda x: x['Total_GPU'], reverse=True)
            try:
                print('nodes', sorted_available_ips[0]['Hostname'])
                # 添加需要的服务列表
                # development_plan.append(sorted_available_ips[0])
                # 任务分配服务ip
                ti_container_num = generate_unique_number(con_num_ti2image)
                for task_type, items in classified.items():
                    if  task_type == 'text2image' or task_type == 'image2image':
                            
                        for task in items:
                            task['service_ip'] = sorted_available_ips[0]['Hostname']
                            try:
                                task['container_num'] = ti_container_num
                                task['frame'] = develop_text2image_items[0]['frame']
                                task['container'] = get_container_name('default', develop_text2image_items[0]['container'], task_type, ti_container_num) + '-' + str(ti_container_num)
                            except:
                                task['container_num'] = ti_container_num
                                task['frame'] = develop_image2image_items[0]['frame'] 
                                task['container'] = get_container_name('default', develop_image2image_items[0]['container'], task_type, ti_container_num) + '-' + str(ti_container_num) 
                            plan_tasks.append(task)
            except:
                print('没有空闲的计算机可以部署这个模型')
                # 读取全部数据
                for task_type, items in classified.items():
                    if  task_type == 'text2image' or task_type == 'image2image':
                        for task in items:
                            wait_tasks.append(task)
                needs_gpu.append(['stable', 'default'])
                
        else:
            filtered_subitems_nums = []
            # 获取已部署容器的编号
            for item in filtered_subitems:
                container = item.get("container", "")
                if "-" in container:
                    last_part = container.rsplit("-", 1)[-1]
                    if last_part.isdigit():
                        filtered_subitems_nums.append(int(last_part))
                    else:
                        filtered_subitems_nums.append(None)  # 最后一部分不是数字
                else:
                    filtered_subitems_nums.append(None)  # 没有 "-"
            # 添加需要的服务列表
            # development_plan.append(filtered_subitems[0])
            for task_type, items in classified.items():
                if  task_type == 'text2image' or task_type == 'image2image':
                    for task in items:
                        task['service_ip'] = filtered_subitems[0]['Hostname']
                        task['container'] = filtered_subitems[0]['container']
                        task['container_num'] = filtered_subitems_nums[0]
                        task['frame'] = 'default'
    # 需要超过两个模型服务实例      
    elif model_diffusion_number != 0 and model_diffusion_number_t != 0:
        # 查看当前已有的模型
        filtered_subitems = [
            item for item in filtered_query
        if 'text2image' in item.get("pod", "") and 'stable' in item.get("container", "").lower()
        ]
        filtered_subitems = filter_duplicate_pods(filtered_subitems)
        available_ips = available_ip(float(3)*1024 + 1000)

        if acutal_task_counts/model_diffusion_number > max_text2image_nums:
            max_task = int(acutal_task_counts/model_diffusion_number) + 1
        else:
            max_task = max_text2image_nums
        sorted_available_ips = sorted(available_ips, key=lambda x: x['Total_GPU'], reverse=True)

        # 查看是否有已部署的模型
        if len(filtered_subitems) == 0:
            # 查看当前资源能否部署全部服务
            if len(sorted_available_ips) >= model_diffusion_number:
                task_iter = -1
                develop_sorted_available_ips = sorted_available_ips[0:model_diffusion_number]
                # 任务分配服务ip
                ti_container_nums = []
                for index_num in range(0, model_diffusion_number):
                    ti_container_nums.append(generate_unique_number(con_num_ti2image))

                for task_type, items in classified.items():
                    if  task_type == 'text2image' or task_type == 'image2image':
                        for task in items:
                            plan_tasks.append(task)
                            task['service_ip'] = develop_sorted_available_ips[int(task_iter/max_task)]['Hostname']
                            try:
                                task['container_num'] = ti_container_nums[int(task_iter/max_task)]
                                task['frame'] = develop_text2image_items[0]['frame']
                                task['container'] = get_container_name('default', develop_text2image_items[0]['container'] ,task_type, ti_container_nums[int(task_iter/max_task)]) + '-' + str(ti_container_nums[int(task_iter/max_task)]) 
                            except:
                                task['container_num'] = ti_container_nums[int(task_iter/max_task)]
                                task['frame'] = develop_image2image_items[0]['frame']
                                task['container'] = get_container_name('default', develop_image2image_items[0]['container'] ,task_type, ti_container_nums[int(task_iter/max_task)]) + '-' + str(ti_container_nums[int(task_iter/max_task)])
                            task_iter = task_iter + 1 
            else:
                task_iter = 1
                develop_sorted_available_ips = sorted_available_ips
                ti_container_nums = []
                for index_num in range(0, model_diffusion_number):
                    ti_container_nums.append(generate_unique_number(con_num_ti2image))
                # 任务分配服务ip
                for task_type, items in classified.items():
                    if  task_type == 'text2image' or task_type == 'image2image':
                        for task in items:
                            if task_iter < len(develop_sorted_available_ips) * max_task :
                                plan_tasks.append(task)
                                task['service_ip'] = develop_sorted_available_ips[int(task_iter/max_task)]['Hostname']
                                try:
                                    task['container_num'] = ti_container_nums[int(task_iter/max_task)]
                                    task['frame'] = develop_text2image_items[0]['frame']
                                    task['container'] = get_container_name('default', develop_text2image_items[0]['container'] ,task_type, ti_container_nums[int(task_iter/max_task)]) + '-' + str(ti_container_nums[int(task_iter/max_task)])  
                                except:
                                    task['container_num'] = ti_container_nums[int(task_iter/max_task)]
                                    task['frame'] = develop_image2image_items[0]['frame']
                                    task['container'] = get_container_name('default', develop_image2image_items[0]['container'] ,task_type, ti_container_nums[int(task_iter/max_task)]) + '-' + str(ti_container_nums[int(task_iter/max_task)])
                                task_iter = task_iter + 1
                            else:
                                task_iter = task_iter + 1
                                wait_tasks.append(task)
                
                for add_num in range(model_diffusion_number - len(develop_sorted_available_ips)):
                    needs_gpu.append(['stable', 'default'])
        # 优先使用已部署的模型
        else:
            filtered_subitems_nums = []
            # 获取已部署容器的编号
            for item in filtered_subitems:
                container = item.get("container", "")
                if "-" in container:
                    last_part = container.rsplit("-", 1)[-1]
                    if last_part.isdigit():
                        filtered_subitems_nums.append(int(last_part))
                    else:
                        filtered_subitems_nums.append(None)  # 最后一部分不是数字
                else:
                    filtered_subitems_nums.append(None)  # 没有 "-"

            print(filtered_subitems[0]['Hostname'])
            ti_container_nums = []
            for index_num in range(0, model_diffusion_number):
                ti_container_nums.append(generate_unique_number(con_num_ti2image))

            iter_num = gen_img_tasks_number/max_task + 1
            
            if celery_task != None:
                online_task_container = get_container_online_counts(celery_task)
            else:
                online_task_container = {}
            
            # 这些模型服务当前资源都可以部署成功
            if len(sorted_available_ips) >= model_diffusion_number - len(filtered_subitems):
                task_iter = 1
                # if task_type in online_task_container:
                #     task_iter = task_iter + online_task_container[task_type]
                develop_sorted_available_ips = sorted_available_ips
                # 任务分配服务ip
                filtered_sign = 0
                deploy_sign = 0
                copy_used_container_task_number = copy.deepcopy(used_container_task_number)
                total_used = used_task_number_text2image + used_task_number_image2image
                for task_type, items in classified.items():
                    if  task_type == 'text2image' or task_type == 'image2image':
                        for task in items:
                            # 直接用已有的模型已经够了
                            if (len(filtered_subitems)*max_text2image_nums - total_used) >= gen_img_tasks_number:
                                task['service_ip'] = filtered_subitems[filtered_sign]['Hostname']
                                # try:
                                task['container_num'] = filtered_subitems_nums[filtered_sign]
                                task['container'] = filtered_subitems[filtered_sign]['container']
                                task['frame'] = 'default'
                                if filtered_subitems[filtered_sign]['container'] not in copy_used_container_task_number:
                                    copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = 0
                                copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] + 1
                                if copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] > max_text2image_nums :
                                    filtered_sign = filtered_sign + 1
                                temp_task_iter = task_iter
                            else:
                                if int(task_iter) + 1*max_text2image_nums< (len(filtered_subitems)*max_text2image_nums - total_used):
                                    try:
                                        task['service_ip'] = filtered_subitems[filtered_sign]['Hostname']
                                        task['container_num'] = filtered_subitems_nums[filtered_sign]
                                        task['container'] = filtered_subitems[filtered_sign]['container']
                                        task['frame'] = 'default'
                                        if filtered_subitems[filtered_sign]['container'] not in copy_used_container_task_number:
                                            copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = 0
                                        copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] + 1
                                        if copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] > max_text2image_nums:
                                            filtered_sign = filtered_sign + 1
                                        temp_task_iter = task_iter
                                    except:
                                        plan_tasks.append(task)
                                        task['service_ip'] = develop_sorted_available_ips[deploy_sign]['Hostname']
                                        try:
                                            task['container_num'] = ti_container_nums[deploy_sign]
                                            task['frame'] = develop_text2image_items[0]['frame']
                                            task['container'] = get_container_name('default', develop_text2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign]) 
                                            # task['container'] = get_container_name('default', develop_text2image_items[0]['container'], task_type, ti_container_nums[deploy_sign])   
                                        except:
                                            task['container_num'] = ti_container_nums[deploy_sign]
                                            task['frame'] = develop_image2image_items[0]['frame']
                                            task['container'] = get_container_name('default', develop_image2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign])
                                        try:
                                            if  (task_iter - temp_task_iter) % max_text2image_nums == 0 and task_iter != temp_task_iter:
                                                deploy_sign = deploy_sign + 1
                                        except:
                                            temp_task_iter = 0
                                            if  (task_iter - temp_task_iter) % max_text2image_nums == 0 and task_iter != temp_task_iter:
                                                deploy_sign = deploy_sign + 1
                                else:
                                    try:
                                        plan_tasks.append(task)
                                        task['service_ip'] = develop_sorted_available_ips[deploy_sign]['Hostname']
                                        try:
                                            task['container_num'] = ti_container_nums[deploy_sign]
                                            task['frame'] = develop_text2image_items[0]['frame']
                                            task['container'] = get_container_name('default', develop_text2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign])  
                                        except:
                                            task['container_num'] = ti_container_nums[deploy_sign]
                                            task['frame'] = develop_image2image_items[0]['frame']
                                            task['container'] = get_container_name('default', develop_image2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign])
                                        try:
                                            if  (task_iter - temp_task_iter) % max_text2image_nums == 0 and task_iter != temp_task_iter:
                                                deploy_sign = deploy_sign + 1
                                        except:
                                            temp_task_iter = 0
                                            if  (task_iter - temp_task_iter) % max_text2image_nums == 0 and task_iter != temp_task_iter:
                                                deploy_sign = deploy_sign + 1
                                    except:
                                        wait_tasks.append(task)
                            task_iter = task_iter + 1
            else:
                # 资源不够全部部署
                task_iter = 1
                # if task_type in online_task_container:
                #     task_iter = task_iter + online_task_container[task_type]
                develop_sorted_available_ips = sorted_available_ips

                filtered_sign = 0
                deploy_sign = 0
                copy_used_container_task_number = copy.deepcopy(used_container_task_number)
                total_used = used_task_number_text2image + used_task_number_image2image
                # 任务分配服务ip
                for task_type, items in classified.items():
                    if  task_type == 'text2image' or task_type == 'image2image':
                        # 赋予初值
                        temp_task_iter = len(filtered_subitems)*max_text2image_nums - total_used
                        for task in items:
                            print('task_iter', task_iter)
                            if int(task_iter) + 1*max_text2image_nums< (len(filtered_subitems)*max_text2image_nums - total_used):
                                print('filtered_signfiltered_sign', filtered_sign)
                                print('filtered_subitemsfiltered_subitems', filtered_subitems, filtered_sign)
                                task['service_ip'] = filtered_subitems[filtered_sign]['Hostname']
                                task['container_num'] = filtered_subitems_nums[filtered_sign]
                                task['container'] = filtered_subitems[filtered_sign]['container']
                                task['frame'] = 'default'
                                if filtered_subitems[filtered_sign]['container'] not in copy_used_container_task_number:
                                    copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = 0
                                copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] + 1
                                print(copy_used_container_task_number[filtered_subitems[filtered_sign]['container']])
                                if copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] > max_text2image_nums:
                                    filtered_sign = filtered_sign + 1
                                temp_task_iter = task_iter
                            elif int(task_iter - temp_task_iter) + 1<= len(develop_sorted_available_ips)*max_text2image_nums:
                                plan_tasks.append(task)
                                try:
                                    task['service_ip'] = develop_sorted_available_ips[deploy_sign]['Hostname']
                                    try:
                                        task['container_num'] = ti_container_nums[deploy_sign]
                                        task['frame'] = develop_text2image_items[0]['frame']
                                        task['container'] = get_container_name('default', develop_text2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign])  
                                        # task['container'] = get_container_name('default', develop_text2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) 
                                        print('ccc7', develop_text2image_items[0]['container'], task_type, str(ti_container_nums[deploy_sign])) 
                                    except:
                                        task['container_num'] = ti_container_nums[deploy_sign]
                                        task['frame'] = develop_image2image_items[0]['frame']
                                        task['container'] = get_container_name('default', develop_image2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign])
                                        # task['container'] = get_container_name('default', develop_image2image_items[0]['container'], task_type, ti_container_nums[deploy_sign]) 
                                        print('ccc8', develop_image2image_items[0]['container'], task_type, str(ti_container_nums[deploy_sign]))
                                    try:
                                        if  (task_iter - temp_task_iter) % max_text2image_nums == 0 and task_iter != temp_task_iter:
                                            deploy_sign = deploy_sign + 1
                                    except:
                                        print('?????????', task_iter, len(filtered_subitems)*max_text2image_nums, total_used)
                                        temp_task_iter = 0
                                        if  (task_iter - temp_task_iter) % max_text2image_nums == 0 and task_iter != temp_task_iter:
                                            deploy_sign = deploy_sign + 1
                                except:
                                    wait_tasks.append(task)
                            else:
                                wait_tasks.append(task)
                            task_iter = task_iter + 1
                for add_num in range(model_diffusion_number - len(sorted_available_ips) - len(filtered_subitems)):
                    needs_gpu.append(['stable', 'default'])
    
    # 非stable类型模型
    # filtered为需要部署的模型
    
    for sub_model in filtered:
        # task_type
        model_name = sub_model['models']
        model_param = sub_model['container']
        model_number = sub_model['model_number']
        model_frame = sub_model['frame']
        this_task_counts = sub_model['task_counts']
        
        corr_numbers = [1,1,5,10,20,40,60,80]
        # 决策要部署多少模型
        if task_type == 'text2text':
            model_number = 1 + int((this_task_counts + used_task_number_text2text)/max_text2text_nums)
        
        elif task_type == 'image2text':
            model_number = 1 + int((this_task_counts + used_task_number_image2text)/max_image2text_nums)

        elif task_type == 'text2video':
            print('video_num', this_task_counts + used_task_number_text2video)
            # print('sssdd', len(classified['text2video']))
            model_number = int((this_task_counts + used_task_number_text2video)/max_text2video_nums)


        develop_items = [
            item for item in filtered
        if model_name  in item.get("models", "").lower()
        ]
        acutal_task_counts = develop_items[0].get("task_counts")
        print('show_result', develop_items)
        if model_name !='text2image' and model_name !='image2image':

            if model_number == 1 and this_task_counts != 0:
                if model_name != 'text2video':
                    filtered_subitems = [
                        item for item in filtered_query
                    if model_name in item.get("pod", "") and model_param in item.get("container", "").lower()
                    ]
                    filtered_subitems = filter_duplicate_pods(filtered_subitems)
                else:
                    filtered_subitems = [
                        item for item in filtered_query
                    if model_name in item.get("pod", "") and 'cogvideo' in item.get("container", "").lower()
                    ]
                    filtered_subitems = filter_duplicate_pods(filtered_subitems)

                if len(filtered_subitems) == 0:
                    # 当前集群无法提供该服务，需要部署
                    # 在集群找能够满足该服务的服务器，优先推理速度更快的
                    if model_name == 'text2text' and model_frame == 'default':
                        available_ips = available_ip(float(''.join(filter(str.isdigit, model_param)))*2*1024 + 2048)
                        ti_container_num = generate_unique_number(con_num_text2text)
                    elif model_name == 'text2text' and model_frame == 'ollama':
                        available_ips = available_ip(float(''.join(filter(str.isdigit, model_param)))*1024 + 1024)
                        ti_container_num = generate_unique_number(con_num_text2text)
                    elif model_name == 'image2text':
                        available_ips = available_ip(float(''.join(filter(str.isdigit, model_param)))*2*1024 + 2048)
                        available_ips = [
                            ip_info for ip_info in available_ips
                            if ip_info['IP'] not in ['192.168.2.5', '192.168.2.6', '192.168.2.7']
                        ]
                        ti_container_num = generate_unique_number(con_num_image2text)
                    elif model_name == 'text2video':
                        available_ips = available_ip(float(18)*1024 + 2048)
                        ti_container_num = generate_unique_number(con_num_text2video)
                    # print('部署位置', available_ips)
                    
                    sorted_available_ips = sorted(available_ips, key=lambda x: x['Total_GPU'], reverse=True)

                    try:
                        for task_type, items in classified.items():
                            if  task_type == model_name:
                                for task in items:
                                    plan_tasks.append(task)
                                    task['service_ip'] = sorted_available_ips[0]['Hostname']
                                    task['container_num'] = ti_container_num
                                    task['frame'] = develop_items[0]['frame']
                                    task['container'] = get_container_name(develop_items[0]['frame'], develop_items[0]['container'], task_type, ti_container_num)     
                                    print('ddd1', develop_items[0]['container'], task_type, str(ti_container_num)) 

                    except:
                        print('没有空闲的计算机可以部署这个模型')
                        for task_type, items in classified.items():
                            if  task_type == model_name:
                                for task in items:
                                    wait_tasks.append(task)
                        needs_gpu.append([model_param, model_frame])

                else:
                    filtered_subitems_nums = []
                    # 获取已部署容器的编号
                    for item in filtered_subitems:
                        print('debug', item)
                        container = item.get("container", "")
                        if "-" in container:
                            last_part = container.rsplit("-", 1)[-1]
                            if last_part.isdigit():
                                filtered_subitems_nums.append(int(last_part))
                            else:
                                filtered_subitems_nums.append(None)  # 最后一部分不是数字
                        else:
                            filtered_subitems_nums.append(None)  # 没有 "-"

                    for task_type, items in classified.items():
                        if  task_type == model_name:
                            for task in items:
                                task['service_ip'] = filtered_subitems[0]['Hostname']
                                task['container'] = filtered_subitems[0]['container']
                                task['container_num'] = filtered_subitems_nums[0]
                                if model_name == 'text2text':
                                    if 'ollama' in task['container']:
                                        task['frame'] = 'ollama'
                                    else:
                                        task['frame'] = 'default'
                                else:
                                    task['frame'] = 'default'

            # 需要超过两个模型服务实例      
            elif model_number != 0 and this_task_counts != 0:
                # 查看当前已有的模型：develop_items
                if model_name != 'filtered_subitems':
                    filtered_subitems = [
                        item for item in filtered_query
                    if model_name in item.get("pod", "") and model_param in item.get("container", "").lower()
                    ]
                    filtered_subitems = filter_duplicate_pods(filtered_subitems)
                else:
                    filtered_subitems = [
                        item for item in filtered_query
                    if model_name in item.get("pod", "") and 'cogvideo' in item.get("container", "").lower()
                    ]
                    filtered_subitems = filter_duplicate_pods(filtered_subitems)
                # container number
                ti_container_nums = []
                # 查看可部署服务的ip列表
                if model_name == 'text2text' and model_frame == 'default':
                    max_new_tasks = max_text2text_nums   
                    available_ips = available_ip(float(''.join(filter(str.isdigit, model_param)))*2*1024 + 2048)
                    if acutal_task_counts/model_number>max_text2text_nums:
                        max_task = int(acutal_task_counts/model_number) + 1
                    else:
                        max_task = max_text2text_nums
                        
                    
                    for index_num in range(0, model_number):
                        ti_container_nums.append(generate_unique_number(con_num_text2text))
                    # max_task = max_text2text_nums
                elif model_name == 'text2text' and model_frame == 'ollama':
                    max_new_tasks = max_text2text_nums
                    available_ips = available_ip(float(''.join(filter(str.isdigit, model_param)))*1024 + 1024)
                    # max_task = max_text2text_nums
                    if acutal_task_counts/model_number>max_text2text_nums:
                        max_task = int(acutal_task_counts/model_number) + 1
                    else:
                        max_task = max_text2text_nums
                    
                    for index_num in range(0, model_number):
                        ti_container_nums.append(generate_unique_number(con_num_text2text))
                elif model_name == 'image2text':
                    max_new_tasks = max_image2text_nums   
                    available_ips = available_ip(float(''.join(filter(str.isdigit, model_param)))*2*1024 + 2048)
                    available_ips = [
                        ip_info for ip_info in available_ips
                        if ip_info['IP'] not in ['192.168.2.5', '192.168.2.6', '192.168.2.7']
                    ]
                    # max_task = max_image2text_nums
                    if acutal_task_counts/model_number>max_image2text_nums:
                        max_task = int(acutal_task_counts/model_number) + 1
                    else:
                        max_task = max_image2text_nums
                    
                    for index_num in range(0, model_number):
                        ti_container_nums.append(generate_unique_number(con_num_image2text))
                elif model_name == 'text2video':
                    max_new_tasks = max_text2video_nums 
                    available_ips = available_ip(float(18)*1024 + 2048)
                    # max_task = max_text2video_nums
                    if acutal_task_counts/model_number>max_text2video_nums:
                        max_task = int(acutal_task_counts/model_number) + 1
                    else:
                        max_task = max_text2video_nums
                    
                    for index_num in range(0, model_number):
                        ti_container_nums.append(generate_unique_number(con_num_text2video))
                sorted_available_ips = sorted(available_ips, key=lambda x: x['Total_GPU'], reverse=True)

                # 查看是否有已部署的模型
                if len(filtered_subitems) == 0:
                    # 查看当前资源能否部署全部服务
                    print('model_number', model_number, sorted_available_ips)
                    if len(sorted_available_ips) >= model_number:
                        task_iter = 0
                        develop_sorted_available_ips = sorted_available_ips[0:model_number]
                        # 任务分配服务ip
                        for task_type, items in classified.items():
                            if  task_type == model_name:
                                print('itemitem', len(items))
                                for task in items:
                                    try:
                                        print(task_type, 'develop_sorted_available_ips',len(develop_sorted_available_ips), task_iter, max_task )
                                        plan_tasks.append(task)
                                        task['service_ip'] = develop_sorted_available_ips[int(task_iter/max_task)]['Hostname']
                                        # task['container'] = develop_items[0]['container']
                                        # task['container_num'] = ti_container_nums[int(task_iter/max_task)]
                                        # task['frame'] = develop_items[0]['frame']
                                        task['container_num'] = ti_container_nums[int(task_iter/max_task)]
                                        task['frame'] = develop_items[0]['frame']
                                        task['container'] = get_container_name(develop_items[0]['frame'], develop_items[0]['container'], task_type, ti_container_nums[int(task_iter/max_task)]) 
                                        print('ddd2', develop_items[0]['container'], task_type, str(ti_container_nums[int(task_iter/max_task)])) 
                                        task_iter = task_iter + 1
                                    except:
                                        task_iter = task_iter + 1
                                        wait_tasks.append(task)

                    else:
                        print('部署所有模型的机器数不够')
                        task_iter = 0
                        develop_sorted_available_ips = sorted_available_ips
                        # 任务分配服务ip
                        for task_type, items in classified.items():
                            if  task_type == model_name:
                                # items_new = remove_duplicate_tasks(items)
                                for task in items:
                                    try:
                                        if task_iter <= len(develop_sorted_available_ips) * max_task and len(develop_sorted_available_ips) != 0:
                                            plan_tasks.append(task)
                                            task['service_ip'] = develop_sorted_available_ips[int(task_iter/max_task)]['Hostname']
                                            task['container_num'] = ti_container_nums[int(task_iter/max_task)]
                                            task['frame'] = develop_items[0]['frame']
                                            task['container'] = get_container_name(develop_items[0]['frame'], develop_items[0]['container'], task_type, ti_container_nums[int(task_iter/max_task)])    
                                            print('ddd3', develop_items[0]['container'], task_type, str(ti_container_nums[int(task_iter/max_task)])) 
                                            task_iter = task_iter + 1
                                        else:
                                            task_iter = task_iter + 1
                                            wait_tasks.append(task)
                                    except:
                                        task_iter = task_iter + 1
                                        wait_tasks.append(task)
                        
                        for add_num in range(model_number - len(develop_sorted_available_ips)):
                            needs_gpu.append([model_param, model_frame])


                # 优先使用已部署的模型,不能给每个模型分配太多任务，考虑上限
                else:
                    if celery_task != None:
                        online_task_container = get_container_online_counts(celery_task)
                        print('online_task_container', online_task_container)
                    else:
                        online_task_container = {}
                    filtered_subitems_nums = []
                    # 获取已部署容器的编号
                    for item in filtered_subitems:
                        container = item.get("container", "")
                        if "-" in container:
                            last_part = container.rsplit("-", 1)[-1]
                            if last_part.isdigit():
                                filtered_subitems_nums.append(int(last_part))
                            else:
                                filtered_subitems_nums.append(None)  # 最后一部分不是数字
                        else:
                            filtered_subitems_nums.append(None)  # 没有 "-"

                    # 计算实际需要的实例数
                    iter_num = acutal_task_counts/max_task + 1
                    # 这些模型服务当前资源都可以部署成功
                    if len(sorted_available_ips) >= model_number - len(filtered_subitems):
                        task_iter = 1

                        develop_sorted_available_ips = sorted_available_ips

                        filtered_sign = 0
                        deploy_sign = 0
                        # if task_type

                        copy_used_container_task_number = copy.deepcopy(used_container_task_number)
                        # 任务分配服务ip
                        for task_type, items in classified.items():
                            if  task_type == model_name:
                                if model_name == 'text2text':
                                    total_used = used_task_number_text2text
                                elif model_name == 'image2text':
                                    total_used = used_task_number_image2text
                                elif model_name == 'text2video':
                                    total_used = used_task_number_text2video
                                for task in items:
    
                                    if (len(filtered_subitems)*max_new_tasks - total_used) >= this_task_counts:
                                        task['service_ip'] = filtered_subitems[filtered_sign]['Hostname']
                                        task['container_num'] = filtered_subitems_nums[filtered_sign]
                                        task['container'] = filtered_subitems[filtered_sign]['container']
                                        if model_name == 'text2text':
                                            if 'ollama' in task['container']:
                                                task['frame'] = 'ollama'
                                            else: 
                                                task['frame'] = 'default'
                                        else:
                                            task['frame'] = 'default'
                                        if filtered_subitems[filtered_sign]['container'] not in copy_used_container_task_number:
                                            copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = 0
                                        copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] + 1
                                        if copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] > max_new_tasks + 1:
                                            filtered_sign = filtered_sign + 1
                                        
                                        temp_task_iter = task_iter
                                    else:
                                        if int(task_iter) + 1*max_new_tasks< (len(filtered_subitems)*max_new_tasks - total_used):
                                            try:
                                                task['service_ip'] = filtered_subitems[filtered_sign]['Hostname']
                                                task['container_num'] = filtered_subitems_nums[filtered_sign]
                                                task['container'] = filtered_subitems[filtered_sign]['container']
                                                if model_name == 'text2text':
                                                    if 'ollama' in task['container']:
                                                        task['frame'] = 'ollama'
                                                    else: 
                                                        task['frame'] = 'default'
                                                else:
                                                    task['frame'] = 'default'
                                                if filtered_subitems[filtered_sign]['container'] not in copy_used_container_task_number:
                                                    copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = 0
                                                copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] + 1
                                                if copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] > max_new_tasks + 1:
                                                    filtered_sign = filtered_sign + 1
                                                temp_task_iter = task_iter
                                            except:
                                                plan_tasks.append(task)
                                                
                                                try:
                                                    task['service_ip'] = develop_sorted_available_ips[deploy_sign]['Hostname']
                                                    task['container_num'] = ti_container_nums[deploy_sign]
                                                    task['frame'] = develop_items[0]['frame']
                                                    # task['container'] = get_container_name('default', develop_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign])  
                                                    task['container'] = get_container_name('default', develop_items[0]['container'], task_type, ti_container_nums[deploy_sign])                                                    try:
                                                        if  (task_iter - temp_task_iter) % max_new_tasks == 0 and task_iter != temp_task_iter:
                                                            deploy_sign = deploy_sign + 1
                                                    except:
                                                        temp_task_iter = 0
                                                        print('?????????max_new_tasks', task_iter, len(filtered_subitems)*max_new_tasks, total_used)
                                                        if  (task_iter - temp_task_iter) % max_new_tasks == 0 and task_iter != temp_task_iter:
                                                            deploy_sign = deploy_sign + 1
                                                except:
                                                    wait_tasks.append(task)
                                        else:
                                            plan_tasks.append(task)
                                        
                                            try:
                                                task['service_ip'] = develop_sorted_available_ips[deploy_sign]['Hostname']
                                                task['container_num'] = ti_container_nums[deploy_sign]
                                                task['frame'] = develop_items[0]['frame']
                                                # task['container'] = get_container_name('default', develop_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign])  
                                                task['container'] = get_container_name('default', develop_items[0]['container'], task_type, ti_container_nums[deploy_sign]) 
                                                print('ddd5', develop_items[0]['container'], task_type, str(ti_container_nums[deploy_sign])) 
                                                try:
                                                    if  (task_iter - temp_task_iter) % max_new_tasks == 0 and task_iter != temp_task_iter:
                                                        deploy_sign = deploy_sign + 1
                                                except:
                                                    temp_task_iter = 0
                                                    print('?????????max_new_tasks', task_iter, len(filtered_subitems)*max_new_tasks, total_used)
                                                    if  (task_iter - temp_task_iter) % max_new_tasks == 0 and task_iter != temp_task_iter:
                                                        deploy_sign = deploy_sign + 1
                                            except:
                                                wait_tasks.append(task)
                                    task_iter = task_iter + 1


                    else:
                        # 资源不足：
                        task_iter = 1
                        if task_type in online_task_container:
                            task_iter = task_iter + online_task_container[task_type]
                        develop_sorted_available_ips = sorted_available_ips
                                                
                        filtered_sign = 0
                        deploy_sign = 0
                        copy_used_container_task_number = copy.deepcopy(used_container_task_number)
                        # 任务分配服务ip
                        for task_type, items in classified.items():
                            if  task_type == model_name:
                                if  task_type == model_name:
                                    if model_name == 'text2text':
                                        total_used = used_task_number_text2text
                                    elif model_name == 'image2text':
                                        total_used = used_task_number_image2text
                                    elif model_name == 'text2video':
                                        total_used = used_task_number_text2video
                                
                                # 赋予初值
                                temp_task_iter = len(filtered_subitems)*max_new_tasks - total_used
                                for task in items:
                                    models_cha = iter_num - len(filtered_subitems)
                                    if int(task_iter) + 1*max_new_tasks< (len(filtered_subitems)*max_new_tasks - total_used):
                                        task['service_ip'] = filtered_subitems[filtered_sign]['Hostname']
                                        task['container_num'] = filtered_subitems_nums[filtered_sign]
                                        task['container'] = filtered_subitems[filtered_sign]['container']
                                        if model_name == 'text2text':
                                            if 'ollama' in task['container']:
                                                task['frame'] = 'ollama'
                                            else: 
                                                task['frame'] = 'default'
                                        else:
                                            task['frame'] = 'default'
                                        if filtered_subitems[filtered_sign]['container'] not in copy_used_container_task_number:
                                            copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = 0
                                        copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] = copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] + 1
                                        if copy_used_container_task_number[filtered_subitems[filtered_sign]['container']] > max_new_tasks +1:
                                            filtered_sign = filtered_sign + 1
                                        temp_task_iter = task_iter
                                    elif int(task_iter - temp_task_iter) + 1<= len(develop_sorted_available_ips)*max_new_tasks:
                                        plan_tasks.append(task)
                                        
                                        try:
                                            task['service_ip'] = develop_sorted_available_ips[deploy_sign]['Hostname']
                                            task['container_num'] = ti_container_nums[deploy_sign]
                                            # task['container'] = get_container_name('default', develop_items[0]['container'], task_type, ti_container_nums[deploy_sign]) + '-' + str(ti_container_nums[deploy_sign]) 
                                            task['container'] = get_container_name('default', develop_items[0]['container'], task_type, ti_container_nums[deploy_sign])  
                                            if model_name == 'text2text':
                                                if 'ollama' in task['container']:
                                                    task['frame'] = 'ollama'
                                                else: 
                                                    task['frame'] = 'default'
                                            else:
                                                task['frame'] = 'default'
                                            try:
                                                if  (task_iter - temp_task_iter) % max_new_tasks == 0 and task_iter != temp_task_iter:
                                                    deploy_sign = deploy_sign + 1
                                            except:
                                                print('?????????max_new_tasks', task_iter, len(filtered_subitems)*max_new_tasks, total_used)
                                                temp_task_iter = 0
                                                if  (task_iter - temp_task_iter) % max_new_tasks == 0 and task_iter != temp_task_iter:
                                                    deploy_sign = deploy_sign + 1
                                        except:
                                            wait_tasks.append(task)
                                    else:
                                        wait_tasks.append(task)
                                    task_iter = task_iter + 1



                        for add_num in range(model_diffusion_number - len(sorted_available_ips) - len(filtered_subitems)):
                            needs_gpu.append([[model_param, model_frame]])


    # print('development_plan', development_plan)
    tasks_type = ['text2text', 'image2text', 'text/image2image', 'text2video']
    
    save_classified_to_timestamped_file(classified)
    # 插入数据到redis

    sync_save_tasks(classified)

    if len(wait_tasks)!= 0:
        # print('wait_tasks', wait_tasks)
        # 创建保存目录（如果不存在）
        os.makedirs(save_wait_dir, exist_ok=True)
        # 逐个保存任务,下一次程序启动时执行
        for task in wait_tasks:
            # 生成文件名（使用 task_id 作为文件名）
            # if task['task_id'] == '2a271b55-3b1c-48dc-928e-12d08a12a259':
            #     print('wawawawawa')
            filename = os.path.join(save_wait_dir, f"{task['task_id']}.json")
            
            # 写入 JSON 文件（含格式化）
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(task, f, ensure_ascii=False, indent=4)

        # print('needs_gpu', needs_gpu)
        
        all_data = load_all_classified()
        if all_data:
            print(f"共加载 {len(all_data)} 个文件")
        result = calculate_non_pending_percentage(all_data)
        # print('查看任务完成情况', result)

        low_percent = find_low_percentage_entries(result, 15)
        # 存在total_tasks占比小于总total_tasks30-40%, 这个模型用完就同时删除这部分权重
        for category, info in low_percent.items():
            print(f"\n分类 {category.upper()}")
            print(f"总任务数: {info['total_tasks']}")
            print("符合条件条目:")
            for entry in info['matched_entries']:
                print(f"  - IP: {entry['service_ip']} | 容器: {entry['container']} | 任务数: {entry['total_tasks']} | 占比: {entry['percentage']}%")

    # 部署任务
    print('develop service')
    unique_data = extract_unique_fields(classified)

    develop_results = {task_type: items[0] for task_type, items in classified.items() if items}

    for developer in unique_data:
        task_type = developer['task_type']
    # for task_type, developer in develop_results.items():
        parser = argparse.ArgumentParser()
        print('部署', task_type, developer)
        # 未给出部署方案的任务
        if developer['container'] == None:
            continue

        if task_type == "image2image":
            task_type = "text2image"

        if developer['frame'] != 'ollama':
            try:
                par_xx = developer['container_num']
            except:
                # 已经部署的模型
                par_xx = developer['container'].split('-')[-1]
            par_input = task_type + "-x" +".yaml"
            # par_input = task_type + "-" + str(par_xx) + ".yaml"
            par_output = './deploy_list/' + task_type + "-" + str(par_xx) + "-" + "development" + ".yaml"
            par_service_ip = developer['service_ip']
            if task_type == 'text2text':
                
                if  '3b' in developer['container']:
                    par_model = "Qwen2.5-3B-Instruct"
                elif '7b' in developer['container']:
                    develop_available_ips = available_ip(double_text2text_params[2])
                    develop_sorted_available_ips = sorted(develop_available_ips, key=lambda x: x['Total_GPU'], reverse=True)
                    hostnames = [node['Hostname'] for node in develop_sorted_available_ips]
                    if par_service_ip in hostnames:
                        par_model = "Qwen2.5-7B-Instruct"
                    else:
                        par_model = "Qwen2.5-3B-Instruct"

                elif '8b' in developer['container']:
                    develop_available_ips = available_ip(double_text2text_params[1])
                    develop_sorted_available_ips = sorted(develop_available_ips, key=lambda x: x['Total_GPU'], reverse=True)
                    hostnames = [node['Hostname'] for node in sorted_available_ips]
                    if par_service_ip in hostnames:
                        par_model = "Meta-Llama-3.1-8B-Instruct"
                    else:
                        par_model = "Qwen2.5-3B-Instruct"
                    # par_model = "Meta-Llama-3.1-8B-Instruct"
                    
                elif '9b' in developer['container']:
                    develop_available_ips = available_ip(double_text2text_params[0])
                    develop_sorted_available_ips = sorted(develop_available_ips, key=lambda x: x['Total_GPU'], reverse=True)
                    hostnames = [node['Hostname'] for node in sorted_available_ips]
                    if par_service_ip in hostnames:
                        par_model = "glm-4-9b-chat"
                    else:
                        par_model = "Qwen2.5-3B-Instruct"

                    # par_model = "glm-4-9b-chat"


            elif task_type == 'image2text':
                if  '2b' in developer['container']:
                    par_model = "Qwen2-VL-2B-Instruct"
                elif '7b' in developer['container']:
                    develop_available_ips = available_ip(double_image2text_params[0])
                    develop_sorted_available_ips = sorted(develop_available_ips, key=lambda x: x['Total_GPU'], reverse=True)
                    hostnames = [node['Hostname'] for node in sorted_available_ips]
                    if par_service_ip in hostnames:
                        par_model = "Qwen2-VL-7B-Instruct"
                    else:
                        par_model = "Qwen2-VL-2B-Instruct"
                    # par_model = "Qwen2-VL-7B-Instruct"
            elif task_type == 'text2image':
                par_model = "stable-diffusion"
            elif task_type == 'text2video':
                par_model = "CogVideoX-2b-sat"

        elif developer['frame'] == 'ollama':
            try:
                par_xx = developer['container_num']
            except:
                # 已经部署的模型
                par_xx = developer['container'].split('-')[-1]

            par_input = task_type + "-" + developer['frame']+ "-x" + ".yaml"
            # par_input = task_type + "-" + developer['frame']+ "-" + str(par_xx) + ".yaml"
            par_output = './deploy_list/' + task_type + "-" + developer['frame'] + "-" + str(par_xx) + "-" + "development" + ".yaml"
            par_service_ip = developer['service_ip']
            if task_type == 'text2text':
                if  '3b' in developer['container']:
                    par_model = "Qwen2.5-3B-ollama"
                elif '7b' in developer['container']:
                    par_model = "Qwen2.5-7B-ollama"
                elif '8b' in developer['container']:
                    par_model = "llama31-8b-ollama"
                elif '9b' in developer['container']:
                    par_model = "glm4-9b-ollama"
            elif task_type == 'image2text':
                if  '2b' in developer['container']:
                    par_model = "Qwen2-VL-2B-Instruct"
                elif '7b' in developer['container']:
                    par_model = "Qwen2-VL-7B-Instruct"  
            elif task_type == 'text2image':
                par_model = "stable-diffusion"
            elif task_type == 'text2video':
                par_model = "CogVideoX-2b-sat"
        
        # 检查本地权重存在情况
        if par_model == "stable-diffusion":
            remote_weight_path = remote_full_path + "yz_diffusion"
            remote_flag_path = remote_full_path + "yz_diffusion" + '_'+ flag
        elif par_model == "CogVideoX-2b-sat":
            remote_weight_path = remote_full_path + "CogVideo-1.0"
            remote_flag_path = remote_full_path + "CogVideo-1.0" + '_'+ flag
        else:
            remote_weight_path = remote_full_path + par_model
            remote_flag_path = remote_full_path + par_model + '_'+ flag
        # print('NODES_IP[par_service_ip], remote_weight_path, ssh_key_path', NODES_IP[par_service_ip], remote_weight_path, ssh_key_path)
        r_weight = check_remote_dir_exists(NODES_IP[par_service_ip], remote_weight_path, ssh_key_path)
        r_flag = check_remote_file_exists(NODES_IP[par_service_ip], remote_flag_path, ssh_key_path)
        print('部署检查', r_weight, r_flag, NODES_IP[par_service_ip], remote_weight_path)
        if r_weight == True and r_flag == True:
            # 本地模型可以用
            par_input = 'local-' + par_input 
            par_output = par_output.replace('./deploy_list/', './deploy_list/local-', 1)

            print('local_result', par_input, par_output)
            parser.add_argument("--input", default=par_input, help="输入模板文件路径")
            parser.add_argument("--output", default=par_output, help="输出文件路径")
            parser.add_argument("--model", default=par_model, help="自定义模型名称")
            parser.add_argument("--node", default=par_service_ip, help="自定义节点名称")
            parser.add_argument("--x", default=par_xx, type=int, help="手动指定x值（可选）")

            args = parser.parse_args()
            print('args.input', args)
            # 执行修改
            if task_type == 'text2text' and developer['frame'] != 'ollama':
                execute_yaml = local_modify_text2text_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )
            elif task_type == 'text2text' and developer['frame'] == 'ollama':
                execute_yaml = local_modify_text2text_ollama_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )
            elif task_type == 'image2text':
                execute_yaml = local_modify_image2text_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )
            elif task_type == 'text2image':
                execute_yaml = local_modify_text2image_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )  
            elif task_type == 'text2video':
                execute_yaml = local_modify_text2video_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )   
        else:  
            # 使用nfs的模型
            print('nfs_result', par_input, par_output)
            parser.add_argument("--input", default=par_input, help="输入模板文件路径")
            parser.add_argument("--output", default=par_output, help="输出文件路径")
            parser.add_argument("--model", default=par_model, help="自定义模型名称")
            parser.add_argument("--node", default=par_service_ip, help="自定义节点名称")
            parser.add_argument("--x", default=par_xx, type=int, help="手动指定x值（可选）")

            args = parser.parse_args()
            print('args.input', args)
            # 执行修改
            if task_type == 'text2text' and developer['frame'] != 'ollama':
                execute_yaml = modify_text2text_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )
            elif task_type == 'text2text' and developer['frame'] == 'ollama':
                execute_yaml = modify_text2text_ollama_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )
            elif task_type == 'image2text':
                execute_yaml = modify_image2text_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )
            elif task_type == 'text2image':
                execute_yaml = modify_text2image_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )  
            elif task_type == 'text2video':
                execute_yaml = modify_text2video_yaml_template(
                    input_file=args.input,
                    output_file=args.output,
                    model_name=args.model,
                    node_name=args.node,
                    x=args.x
                )   


        print('yaml_file', par_output)

        yaml_file = par_output

        try:
            # 安全执行命令（防止命令注入）
            cmd = f"kubectl apply -f {shlex.quote(yaml_file)}"
            
            print(cmd)
            # 同步执行并捕获输出
            result = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                check=True  # 非零返回码时触发CalledProcessError
            )
            
            print("执行成功！")
            print("输出内容：\n", result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"命令执行失败（返回码 {e.returncode}）:")
            print("错误输出：\n", e.stderr)
            
        except FileNotFoundError:
            print("错误：未找到kubectl命令，请确认：")
            print("1. kubectl已安装\n2. 已添加到系统PATH环境变量")
            
        except Exception as e:
            print(f"未知错误: {str(e)}")

    # 用完以后删除yaml文件
    try:
        os.remove(yaml_file)
        print(f"develop_task文件 {yaml_file} 删除成功")
    except FileNotFoundError:
        print(f"错误：文件 {yaml_file} 不存在")
    except PermissionError:
        print(f"错误：无权限删除文件 {yaml_file}")
    except Exception as e:
        print(f"删除失败，未知错误: {str(e)}")

    send_task_list = []
    receiced_task_ids = []
    for sub_task in tasks:
        
        json_folder = "/yaozhi/vLLM-k8s-operator/deployment/deployment_design/tasks_ip"  # 替换为实际路径
        tasks_by_id = process_json_files(json_folder)
        
        check_task_id = sub_task['task_id']
        if check_task_id in tasks_by_id:
            task_records = tasks_by_id[check_task_id]
            # 按文件时间戳排序 (最新的在后)
            task_records.sort(key=lambda r: r["file_timestamp"])
            # print(f"\n任务 '{check_task_id}' 的所有状态记录:")
            for record in task_records:
                status_info = f"状态: {record['status']}, "
                status_info += f"服务IP: {record['service_ip']}, "
                status_info += f"容器: {record['container']}, "
                status_info += f"文件: {record['source_file']}, "
                status_info += f"时间: {record['file_timestamp']}"
                # print(status_info)
            # 获取最新状态
            latest_record = task_records[-1]
            if latest_record['service_ip'] is None or latest_record['container'] is None:
                print(f"\n最新状态: 任务未完成")
                continue

        task_prompt = {
            "task_id": sub_task['task_id'],
            "task_type": sub_task['task_type'],  # 测试任务类型
            "description": sub_task['description'],
            "file_path": sub_task['file'],  # 如需测试文件任务，指定文件路径
            "max_retries": 3,  # 失败重试次数
            "poll_interval": 5,  # 结果轮询间隔（秒）
            "timeout": 600,  # 单任务超时时间（秒）
        }
        send_task_list.append(task_prompt)
    
    if send_task_list != []:
        send_tasks(send_task_list)

    print('重新发送到fastapi上')




def delete_task_by_id(target_task_id, online_tasks, tasks, classified, file_stats):
    # 1. 删除 online_tasks 中的路径
    i = len(online_tasks) - 1
    while i >= 0:
        path = online_tasks[i]
        if path.split("/")[-1].split(".")[0] == target_task_id:
            online_tasks.pop(i)
        i -= 1

    # 2. 删除 tasks 中的任务字典
    deleted_task_type = None
    for task in tasks:
        if task['task_id'] == target_task_id:
            deleted_task_type = task['task_type']
            break
    tasks[:] = [task for task in tasks if task['task_id'] != target_task_id]

    # 3. 删除 classified 中的任务
    for category in classified:
        classified[category][:] = [
            task for task in classified[category]
            if task['task_id'] != target_task_id
        ]

    # 4. 更新 file_stats 计数
    if deleted_task_type and file_stats[deleted_task_type] > 0:
        file_stats[deleted_task_type] -= 1

def find_value(task_type, tasks):
    for task, value in tasks:
        if task == task_type:
            return value
    return None  # 未找到时返回 None

if __name__ == "__main__":

    directory = './task_groups'
    save_wait_dir = './task_wait'
    save_online_task_dir = './task_groups_online'
    total_stats = defaultdict(int)
    file_count = 0

    # 获取目录下所有JSON文件列表
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    total_files = len(files)

    classified = {
            'text2text': [],
            'image2text': [],
            'text2image': [],
            'image2image': [],
            'text2video': []
    }

    classified_avg_time = {
        'text2text': None,
        'image2text': None,
        'text2image': None,
        'image2image': None,
        'text2video': None 
    }

    classified_nums = {
        'text2text': None,
        'image2text': None,
        'text2image': None,
        'image2image': None,
        'text2video': None
    }
    
    optimizer = UXCostOptimizer(init_alpha=1, init_beta=1)

    while 1:
        online_tasks = check_and_get_files([save_online_task_dir, save_wait_dir])
        # print('task_length',  len(online_tasks))
        if online_tasks:
            tasks = []
            file_stats = defaultdict(int)
            # print("加载刚接收的任务：")
            for online_task_file in online_tasks:  # 关键修改点：遍历文件列表
                # print(online_task_file)
                try:
                    with open(online_task_file, 'r', encoding='utf-8') as f:
                        online_task = json.load(f)
                except:
                    print('读取和写入冲突了')
                    time.sleep(0.5)
                # 统计当前文件
                task = online_task
                task_source_file = online_task_file
                task_type = task.get('task_type', 'UNKNOWN')
                file_stats[task_type] += 1
                total_stats[task_type] += 1
                task_create_time_str = task.get('create_time')
                # 添加任务到任务列表里
                task_type = task['task_type']
                if task_type in classified:
                    # 深拷贝任务并添加新字段
                    new_task = task.copy()
                    new_task['task_source_file'] = task_source_file
                    new_task['service_ip'] = None
                    new_task['container'] = None
                    new_task['frame'] = None
                    create_time = datetime.fromisoformat(task_create_time_str) 
                    new_task['create_time'] = create_time
                    new_task['wait_time'] = (datetime.now() - create_time).total_seconds()
                    classified[task_type].append(new_task)
                    tasks.append(task)  # 合并到主任务列表
                else:
                    print(f"警告：发现未定义的任务类型 {task_type}")
        else:
            tasks = []
            file_stats = defaultdict(int)
            
        filtered_tasks = {}
        task_ids = []
        tasks_record = load_latest_classified()
        celery_task = get_non_success_tasks()
        
        # {'stable-diffusion-76': 4, 'qwen2-vl-2b-instruct-77': 4}    

        if celery_task == None:
            # 无任务
            thread_num = 0
            uuid_list = []
        else:
            # 获取未完成正在执行的任务
            thread_num = len(celery_task)
            uuid_list = list(celery_task.keys())
            for uid in uuid_list:
                task_ids.append(str(uid))

            if tasks_record != None:
                for task_type, iotasks in tasks_record.items():
                    # 过滤出匹配的 task_id
                    matched = [task for task in iotasks if task["task_id"] in task_ids]
                    if matched:
                        filtered_tasks[task_type] = matched
        

        if online_tasks :
            json_folder = "/yaozhi/vLLM-k8s-operator/deployment/deployment_design/tasks_ip"  # 替换为实际路径
            tasks_by_id = process_json_files(json_folder)
            # print('thread_num', thread_num)
            if len(online_tasks) <= 110 - thread_num:

                develop_task(online_tasks, file_stats, tasks, classified, celery_task)
                
                
                if len(online_tasks) != 0:
                    for online_task_file in online_tasks:  # 关键修改点：遍历文件列表
                        # time.sleep(100)
                        # 用完以后删除yaml文件
                        print('online_task_file', online_task_file)
                        if 'task_groups_online' not in online_task_file:
                            # task_wait有些任务没完成，不要误删没完成的任务
                            check_task_id = online_task_file.split('/')[-1].split('.')[0]

                            if check_task_id in tasks_by_id:
                                task_records = tasks_by_id[check_task_id]
                                
                                # 按文件时间戳排序 (最新的在后)
                                task_records.sort(key=lambda r: r["file_timestamp"])
                                
                                # print(f"\n任务 '{check_task_id}' 的所有状态记录:")
                                
                                for record in task_records:
                                    status_info = f"状态: {record['status']}, "
                                    status_info += f"服务IP: {record['service_ip']}, "
                                    status_info += f"容器: {record['container']}, "
                                    status_info += f"文件: {record['source_file']}, "
                                    status_info += f"时间: {record['file_timestamp']}"
                                    # print(status_info)
                                    
                                # 获取最新状态
                                latest_record = task_records[-1]
                                if latest_record['service_ip'] is None or latest_record['container'] is None:
                                    print(f"\n最新状态: 任务未找到目的地，不能删除")
                                    continue
                            
                        try:
                            os.remove(online_task_file)
                            print(f"文件 {online_task_file} 删除成功")
                        except FileNotFoundError:
                            print(f"错误：文件 {online_task_file} 不存在")
                        except PermissionError:
                            print(f"错误：无权限删除文件 {online_task_file}")
                        except Exception as e:
                            print(f"删除失败，未知错误: {str(e)}")


                if file_stats['image2text'] + file_stats['text2image'] +  file_stats['text2text'] + file_stats['image2image'] +file_stats['text2video']!=0:
                    remote_send_weight = []
                    nodes_gpu = query_prometheus_gpu()
                    service_content = query_pods_info()
                    send_weight = [item for item in service_content if item['pod'] != '']
                    # print('k8s_service_content', service_content)
                    loss_gpu = []
                    for item in service_content:
                        if 'text2video' in item:
                            loss_gpu.append(item['Hostname'])    

                        # 提取 Pod 名称前缀（第一个 '-' 前的内容）
                        service_type = item['pod'].split('-')[0]
                        
                    for node in nodes_gpu:
                        if node['Hostname'] == 'b410-4090d-2':
                            node['GPU'] = '1'  # 保持字符串类型
                            # break  # 找到后立即退出循环
                        # text2video在的容器提前把显存扣掉
                        if node['Hostname'] in loss_gpu:
                            if node['GPU']>5000:
                                node['GPU'] = node['GPU'] - 19000  # 保持字符串类型
                                if node['GPU'] <0:
                                    node['GPU'] = 0
                    
                    #  过滤掉不需要的节点
                    filtered_nodes = [
                        node for node in nodes_gpu
                        if node['Hostname'] not in ['b410-2070s-4', 'b410-4090d-2', 'b410-4090d-3']
                    ]

                    # 创建节点名到总显存的映射
                    name_to_total_gpu = {info['Name']: info['GPU'] for info in NODES_INI.values()}

                    # 计算使用百分比并按从高到低排序
                    sorted_nodes_by_usage = sorted(
                        filtered_nodes,
                        key=lambda node: (
                            (name_to_total_gpu[node['Hostname']] - int(node['GPU'])) / 
                            name_to_total_gpu[node['Hostname']]
                        ),
                        reverse=False
                    )
                    sorted_nodes_gpu = sorted_nodes_by_usage
                    online_text2video = sum(1 for item in send_weight if "text2video" in item["pod"])
                    online_image2text = sum(1 for item in send_weight if "image2text" in item["pod"])
                    online_text2text = sum(1 for item in send_weight if "text2text" in item["pod"])
                    online_text2image = sum(1 for item in send_weight if "text2image" in item["pod"])
                    online_image2image = sum(1 for item in send_weight if "image2image" in item["pod"])
                    
                    items_weight = list(file_stats.items())

                    # 按值排序（降序）
                    sorted_items_weight = sorted(items_weight, key=lambda x: x[1], reverse=True)

                    # 取前两个键
                    top_two_keys = [item[0] for item in sorted_items_weight[:2]]
                    print(top_two_keys)  # 输出: ['text2video', 'text2text']
                    
                    if top_two_keys[0] == 'text2video' or top_two_keys[1] == 'text2video':
                        new_sorted_nodes_gpu = copy.deepcopy(sorted_nodes_gpu)

                    if top_two_keys[0] == 'image2text' or top_two_keys[1] == 'image2text':

                        try:
                            new_sorted_nodes_gpu2 = copy.deepcopy(new_sorted_nodes_gpu)
                        except:
                            # 没有text2video
                            new_sorted_nodes_gpu2 = copy.deepcopy(sorted_nodes_gpu)

                        sorted_params = sorted(enumerate(weight_image2text_params), key=lambda x: x[1])
                        print(sorted_params)
                        # 将节点按 GPU 值降序排序
                        # 按参数优先级查找符合条件的节点
                        result = None
                        for param_index, param_value in sorted_params:
                            for host in new_sorted_nodes_gpu2:
                                if host['Hostname'] == 'b410-2070s-1' or host['Hostname'] == 'b410-2070s-2' or host['Hostname'] == 'b410-2070s-3':
                                    continue
                                if int(host['GPU']) > param_value:
                                    if host['Hostname'] == 'b410-4090d-3':
                                        # nfs服务器不需要传输
                                        continue
                                    for new_host in new_sorted_nodes_gpu2:
                                        if new_host['Hostname'] == host['Hostname']:
                                            new_host['GPU'] = int(new_host['GPU']) - param_value
                                            break
                                    result = {
                                        "model_type": 'image2text',
                                        "param_index": param_index,
                                        "param_value": param_value,
                                        "hostname": host['Hostname'],
                                        "gpu_value": int(host['GPU']),
                                        "frame": "default"
                                    }
                                    print('param_value', param_value)
                                    remote_send_weight.append([host['Hostname'], param_path['image2text'][param_value]])
                                    break

                    if top_two_keys[0] == 'text2image' or top_two_keys[1] == 'text2image' or top_two_keys[0] == 'image2image' or top_two_keys[1] == 'image2image':
                    # if top_two_keys[0] == 'image2text' or top_two_keys[1] == 'image2text':
                        try:
                            new_sorted_nodes_gpu3 = copy.deepcopy(new_sorted_nodes_gpu2)
                        except:
                            try:
                                # 有text2video
                                new_sorted_nodes_gpu3 = copy.deepcopy(new_sorted_nodes_gpu)
                            except:
                                # 没有text2video
                                new_sorted_nodes_gpu3 = copy.deepcopy(sorted_nodes_gpu)

                        sorted_params = sorted(enumerate(weight_text2image_params), key=lambda x: x[1])
                        result = None
                        for param_index, param_value in sorted_params:
                            for host in new_sorted_nodes_gpu3:

                                if int(host['GPU']) > param_value:
                                    if host['Hostname'] == 'b410-4090d-3':
                                        # nfs服务器不需要传输
                                        continue
                                    for new_host in new_sorted_nodes_gpu3:
                                        if new_host['Hostname'] == host['Hostname']:
                                            new_host['GPU'] = int(new_host['GPU']) - param_value
                                            break
                                    result = {
                                        "model_type": 'text2image',
                                        "param_index": param_index,
                                        "param_value": param_value,
                                        "hostname": host['Hostname'],
                                        "gpu_value": int(host['GPU']),
                                        "frame": "default"
                                    }
                                    remote_send_weight.append([host['Hostname'], param_path['text2image'][weight_text2image_params[0]]])
                                    break
                            if result:  # 如果找到结果，跳出外层循环
                                break

                    if top_two_keys[0] == 'text2text' or top_two_keys[1] == 'text2text':
                        try:
                            # 有text2text
                            new_sorted_nodes_gpu4 = copy.deepcopy(new_sorted_nodes_gpu3)
                        except:
                            try:
                                try:
                                    # 有image2text
                                    new_sorted_nodes_gpu4 = copy.deepcopy(new_sorted_nodes_gpu2)
                                except:
                                    # 有text2video
                                    new_sorted_nodes_gpu4 = copy.deepcopy(new_sorted_nodes_gpu)
                            except:
                                # 没有text2video
                                new_sorted_nodes_gpu4 = copy.deepcopy(sorted_nodes_gpu)

                        # 将 text2text_params 按值降序排序 (保留原索引)
                        sorted_params = sorted(enumerate(weight_text2text_params), key=lambda x: x[1])
                        # 将节点按 GPU 值降序排序
                        # 按参数优先级查找符合条件的节点
                        result = None
                        for param_index, param_value in sorted_params:
                            for host in new_sorted_nodes_gpu4:
                                if host['Hostname'] == 'b410-4090d-3':
                                    # nfs服务器不需要传输
                                    continue
                                for new_host in new_sorted_nodes_gpu4:
                                    if new_host['Hostname'] == host['Hostname']:
                                        new_host['GPU'] = int(new_host['GPU']) - param_value/2
                                        break
                                result = {
                                    "model_type": 'text2text',
                                    "param_index": param_index,
                                    "param_value": param_value,
                                    "hostname": host['Hostname'],
                                    "gpu_value": int(host['GPU']),
                                    "frame": "ollama"
                                }
                                remote_send_weight.append([host['Hostname'], param_path['text2text'][param_value/2]])
                                break
                            if result:  # 如果找到结果，跳出外层循环
                                break


                print('send_weight', remote_send_weight)
                merge_and_transfer_lists([], remote_send_weight)


            elif 110 - thread_num > 30:
                print('thread is not enough')
                # 剩余线程大于30再开启
                # 现有的程序不能全部完成这些线程, 要做的任务列表需要重新定义一下
                pro_online_tasks = []
                pro_file_stats = defaultdict(int)
                pro_tasks = []
                pro_classified = {
                    'text2text': [],
                    'image2text': [],
                    'text2image': [],
                    'image2image': [],
                    'text2video': []
                }
                # online_tasks需要进行筛选的是

                for task_type, items in classified.items():
                    classified_nums[task_type] = len(items)
                    wait_times = [item["wait_time"] for item in items if "wait_time" in item]
                    if wait_times:
                        classified_avg_time[task_type] = sum(wait_times) / len(wait_times)
                
                sorted_classified_nums = dict(sorted(
                    classified_nums.items(),
                    key=lambda item: item[1],
                    reverse=True
                ))

                # 输出排序后的结果
                print(sorted_classified_nums)
                # max_type_num = next(iter(sorted_classified_nums.values()))
                max_type_num = next(iter(sorted_classified_nums))
                num_available_thread = 110 - thread_num 


                # 排序逻辑
                sorted_items = sorted(
                    classified_avg_time.items(),
                    key=lambda x: (x[1] is not None, x[1]),  # 元组排序键
                    reverse=True  # 反转排序方向
                )
                
                # 生成有序字典
                sorted_classified_avg_time = dict(sorted_items)

                filtered_items_avg_time = [
                    (key, value) 
                    for key, value in sorted_classified_avg_time.items() 
                    if value is not None
                ]
                # print('filtered_items_avg_time', filtered_items_avg_time)
                filtered_result_items_avg_time = [item for item in filtered_items_avg_time if item[0] != 'text2video']
                first_value = filtered_result_items_avg_time[0][1]
                second_value = find_value(max_type_num, filtered_result_items_avg_time)

                filterted_classified_nums = {k: v for k, v in classified_nums.items() if v != 0}


                temp_task = []

                FIFO_classified = process_all_tasks(classified)
                
                if num_available_thread - len(FIFO_classified) <= 0:
                    use_task_lists = FIFO_classified[:num_available_thread]
                else:
                    use_task_lists = FIFO_classified
                    filterted_classified_nums.pop(filtered_result_items_avg_time[0][0], None)
                    classified_avg_time.pop(filtered_result_items_avg_time[0][0], None) 

                print('use_task_lists', use_task_lists)
                for sub_content in use_task_lists:  # 关键修改点：遍历文件列表
                    # 统计当前文件
                    analy_task_id = sub_content['task_id']
                    analy_task_type = sub_content['task_type']
                    temp_task.append(analy_task_id)
                    pro_file_stats[analy_task_type] += 1

                add_pro_tasks = [
                    task for task in tasks
                    if task["task_id"] in temp_task
                ]

                for i in add_pro_tasks:
                    pro_tasks.append(i)

                # --------------------------------
                # 2. 过滤分类字典 classified
                # --------------------------------
                add_pro_classified = {
                    category: [
                        task for task in readtasks
                        if task["task_id"] in temp_task
                    ]
                    for category, readtasks in classified.items()
                }

                for category, class_tasks in add_pro_classified.items():
                    pro_classified[category].extend(class_tasks)

                # --------------------------------
                # 3. 过滤在线任务路径 online_tasks
                # --------------------------------
                add_pro_online_tasks = [
                    path for path in online_tasks
                    if path.split("/")[-1].split(".")[0] in temp_task
                ]
                for i in add_pro_online_tasks:
                    pro_online_tasks.append(i)
                    
                # print('temp_task_show',temp_task)

                print('online_tasks前', len(online_tasks))
                # 从四个列表里删除指定的id的内容
                print('temp_task', temp_task)
                for delete_id in temp_task:
                    # temp_task
                    # 调用示例
                    delete_task_by_id(
                        target_task_id=delete_id,
                        online_tasks=online_tasks,
                        tasks=tasks,
                        classified=classified,
                        file_stats=file_stats
                    )
                print('online_tasks后', len(online_tasks))

                print('classified_avg_time3', classified_avg_time)
                for task_type, items in classified.items():
                    # if len(items) != 0 :
                    #     filterted_classified_nums[task_type] = len(items)
                    wait_times = [item["wait_time"] for item in items if "wait_time" in item]
                    if wait_times:
                        classified_avg_time[task_type] = sum(wait_times) / len(wait_times)
                

                # 删除classified_avg_time中的指定type

                sorted_classified_nums = dict(sorted(
                    filterted_classified_nums.items(),
                    key=lambda item: item[1],
                    reverse=True
                ))

                try:
                    # 输出排序后的结果
                    print(sorted_classified_nums)
                    max_type_num = next(iter(sorted_classified_nums))
                    # 排序逻辑
                    sorted_items = sorted(
                        classified_avg_time.items(),
                        key=lambda x: (x[1] is not None, x[1]),  # 元组排序键
                        reverse=True  # 反转排序方向
                    )
                    
                    # 生成有序字典
                    sorted_classified_avg_time = dict(sorted_items)

                    filtered_items_avg_time = [
                        (key, value) 
                        for key, value in sorted_classified_avg_time.items() 
                        if value is not None
                    ]
                    filtered_result_items_avg_time = [item for item in filtered_items_avg_time if item[0] != 'text2video']
                    first_value = filtered_result_items_avg_time[0][1]
                    second_value = find_value(max_type_num, filtered_result_items_avg_time)
                except:
                    print('???')

                develop_task(pro_online_tasks, pro_file_stats, pro_tasks, pro_classified, celery_task)

                if len(pro_online_tasks) != 0:
                    json_folder = "/yaozhi/vLLM-k8s-operator/deployment/deployment_design/tasks_ip"  # 替换为实际路径
                    tasks_by_id = process_json_files(json_folder)
                    for online_task_file in pro_online_tasks:  # 关键修改点：遍历文件列表
                        check_task_id = online_task_file.split('/')[-1].split('.')[0]
                        if check_task_id in tasks_by_id:
                            task_records = tasks_by_id[check_task_id]
                            
                            # 按文件时间戳排序 (最新的在后)
                            task_records.sort(key=lambda r: r["file_timestamp"])
                            
                            # print(f"\n任务 '{check_task_id}' 的所有状态记录:")
                            
                            for record in task_records:
                                status_info = f"状态: {record['status']}, "
                                status_info += f"服务IP: {record['service_ip']}, "
                                status_info += f"容器: {record['container']}, "
                                status_info += f"文件: {record['source_file']}, "
                                status_info += f"时间: {record['file_timestamp']}"
                                # print(status_info)
                                
                            # 获取最新状态
                            latest_record = task_records[-1]
                            if latest_record['service_ip'] is None or latest_record['container'] is None:
                                print(f"\n最新状态: pro任务未完成, 不能删除")
                                continue
                            # else:
                            #     print(f"\n最新状态: pro任务已完成, 可以删除, 容器: {latest_record['container']}")

                        try:
                            os.remove(online_task_file)
                            print(f"文件 {online_task_file} 删除成功")
                        except FileNotFoundError:
                            print(f"错误：文件 {online_task_file} 不存在")
                        except PermissionError:
                            print(f"错误：无权限删除文件 {online_task_file}")
                        except Exception as e:
                            print(f"删除失败，未知错误: {str(e)}")

                        
                print('file_stats', file_stats)
                if file_stats['image2text'] + file_stats['text2image'] +  file_stats['text2text'] + file_stats['image2image'] +file_stats['text2video']!=0:
                    remote_send_weight = []
                    nodes_gpu = query_prometheus_gpu()
                    service_content = query_pods_info()
                    send_weight = [item for item in service_content if item['pod'] != '']
                    # print('k8s_service_content', service_content)
                    loss_gpu = []
                    for item in service_content:
                        if 'text2video' in item:
                            loss_gpu.append(item['Hostname'])    

                        # 提取 Pod 名称前缀（第一个 '-' 前的内容）
                        service_type = item['pod'].split('-')[0]
                        
                    for node in nodes_gpu:
                        if node['Hostname'] == 'b410-4090d-2':
                            node['GPU'] = '1'  # 保持字符串类型
                            # break  # 找到后立即退出循环
                        # text2video在的容器提前把显存扣掉
                        if node['Hostname'] in loss_gpu:
                            if node['GPU']>5000:
                                node['GPU'] = node['GPU'] - 19000  # 保持字符串类型
                                if node['GPU'] <0:
                                    node['GPU'] = 0
                    

                    sorted_nodes_gpu = sorted(nodes_gpu, key=lambda x: int(x['GPU']), reverse=True)

                    online_text2video = sum(1 for item in send_weight if "text2video" in item["pod"])
                    online_image2text = sum(1 for item in send_weight if "image2text" in item["pod"])
                    online_text2text = sum(1 for item in send_weight if "text2text" in item["pod"])
                    online_text2image = sum(1 for item in send_weight if "text2image" in item["pod"])
                    online_image2image = sum(1 for item in send_weight if "image2image" in item["pod"])

                    if file_stats['text2video'] != 0 and online_text2video != []:
                        
                        new_sorted_nodes_gpu = copy.deepcopy(sorted_nodes_gpu)
                        # 将 text2text_params 按值降序排序 (保留原索引)
                        sorted_params = sorted(enumerate(weight_text2video_params), key=lambda x: x[1])
                        result = None
                        for param_index, param_value in sorted_params:
                            for host in sorted_nodes_gpu:
                                if int(host['GPU']) > param_value:
                                    if host['Hostname'] == 'b410-4090d-3':
                                        # nfs服务器不需要传输
                                        continue
                                    for new_host in new_sorted_nodes_gpu:
                                        if new_host['Hostname'] == host['Hostname']:
                                            new_host['GPU'] = int(new_host['GPU']) - param_value
                                            break
                                    result = {
                                        "model_type": 'text2video',
                                        "param_index": param_index,
                                        "param_value": param_value,
                                        "hostname": host['Hostname'],
                                        "gpu_value": int(host['GPU']),
                                        "frame": "default"
                                    }
                                    remote_send_weight.append([host['Hostname'], param_path['text2video'][weight_text2video_params[0]]])
                                    break
                            if result:  # 如果找到结果，跳出外层循环
                                break

                    if file_stats['image2text'] != 0 and online_image2text != []:
                        
                        try:
                            new_sorted_nodes_gpu2 = copy.deepcopy(new_sorted_nodes_gpu)
                        except:
                            # 没有text2video
                            new_sorted_nodes_gpu2 = copy.deepcopy(sorted_nodes_gpu)

                        sorted_params = sorted(enumerate(weight_image2text_params), key=lambda x: x[1])
                        result = None
                        for param_index, param_value in sorted_params:
                            for host in new_sorted_nodes_gpu2:
                                if host['Hostname'] == 'b410-2070s-1' or host['Hostname'] == 'b410-2070s-2' or host['Hostname'] == 'b410-2070s-3':
                                    continue
                                if int(host['GPU']) > param_value:
                                    if host['Hostname'] == 'b410-4090d-3':
                                        # nfs服务器不需要传输
                                        continue
                                    for new_host in new_sorted_nodes_gpu2:
                                        if new_host['Hostname'] == host['Hostname']:
                                            new_host['GPU'] = int(new_host['GPU']) - param_value
                                            break
                                    result = {
                                        "model_type": 'image2text',
                                        "param_index": param_index,
                                        "param_value": param_value,
                                        "hostname": host['Hostname'],
                                        "gpu_value": int(host['GPU']),
                                        "frame": "default"
                                    }
                                    print('param_value', param_value)
                                    remote_send_weight.append([host['Hostname'], param_path['image2text'][param_value]])
                                    break
                            if result:  # 如果找到结果，跳出外层循环
                                break
                    if file_stats['text2text'] != 0 and online_text2text != []:

                        try:
                            new_sorted_nodes_gpu3 = copy.deepcopy(new_sorted_nodes_gpu2)
                        except:
                            try:
                                # 有text2video
                                new_sorted_nodes_gpu3 = copy.deepcopy(new_sorted_nodes_gpu)
                            except:
                                # 没有text2video
                                new_sorted_nodes_gpu3 = copy.deepcopy(sorted_nodes_gpu)

                        # 将 text2text_params 按值降序排序 (保留原索引)
                        sorted_params = sorted(enumerate(weight_text2text_params), key=lambda x: x[1])
                        # 将节点按 GPU 值降序排序
                        # 按参数优先级查找符合条件的节点
                        result = None
                        for param_index, param_value in sorted_params:
                            for host in new_sorted_nodes_gpu3:
                                if int(host['GPU']) > param_value:
                                    if host['Hostname'] == 'b410-4090d-3':
                                        # nfs服务器不需要传输
                                        continue
                                    for new_host in new_sorted_nodes_gpu3:
                                        if new_host['Hostname'] == host['Hostname']:
                                            new_host['GPU'] = int(new_host['GPU']) - param_value
                                            break
                                    result = {
                                        "model_type": 'text2text',
                                        "param_index": param_index,
                                        "param_value": param_value,
                                        "hostname": host['Hostname'],
                                        "gpu_value": int(host['GPU']),
                                        "frame": "default"
                                    }
                                    remote_send_weight.append([host['Hostname'], param_path['text2text'][param_value]])
                                    break
                                elif int(host['GPU']) > param_value/2:
                                    if host['Hostname'] == 'b410-4090d-3':
                                        # nfs服务器不需要传输
                                        continue
                                    for new_host in new_sorted_nodes_gpu3:
                                        if new_host['Hostname'] == host['Hostname']:
                                            new_host['GPU'] = int(new_host['GPU']) - param_value/2
                                            break
                                    result = {
                                        "model_type": 'text2text',
                                        "param_index": param_index,
                                        "param_value": param_value,
                                        "hostname": host['Hostname'],
                                        "gpu_value": int(host['GPU']),
                                        "frame": "ollama"
                                    }
                                    remote_send_weight.append([host['Hostname'], param_path['text2text'][param_value/2]])
                                    break
                            if result:  # 如果找到结果，跳出外层循环
                                break

                    if (file_stats['text2image'] != 0 and online_text2image != []) or (file_stats['image2image'] != 0 and online_image2image != []):
                        try:
                            # 有text2text
                            new_sorted_nodes_gpu4 = copy.deepcopy(new_sorted_nodes_gpu3)
                        except:
                            try:
                                try:
                                    # 有image2text
                                    new_sorted_nodes_gpu4 = copy.deepcopy(new_sorted_nodes_gpu2)
                                except:
                                    # 有text2video
                                    new_sorted_nodes_gpu4 = copy.deepcopy(new_sorted_nodes_gpu)
                            except:
                                # 没有text2video
                                new_sorted_nodes_gpu4 = copy.deepcopy(sorted_nodes_gpu)


                        # 将 text2text_params 按值降序排序 (保留原索引)
                        sorted_params = sorted(enumerate(weight_text2image_params), key=lambda x: x[1])
                        result = None
                        for param_index, param_value in sorted_params:
                            for host in new_sorted_nodes_gpu4:

                                if int(host['GPU']) > param_value:
                                    if host['Hostname'] == 'b410-4090d-3':
                                        # nfs服务器不需要传输
                                        continue
                                    for new_host in new_sorted_nodes_gpu4:
                                        if new_host['Hostname'] == host['Hostname']:
                                            new_host['GPU'] = int(new_host['GPU']) - param_value
                                            break
                                    result = {
                                        "model_type": 'text2image',
                                        "param_index": param_index,
                                        "param_value": param_value,
                                        "hostname": host['Hostname'],
                                        "gpu_value": int(host['GPU']),
                                        "frame": "default"
                                    }
                                    remote_send_weight.append([host['Hostname'], param_path['text2image'][weight_text2image_params[0]]])
                                    break
                            if result:  # 如果找到结果，跳出外层循环
                                break
                
                print('send_weight', remote_send_weight)
                merge_and_transfer_lists([], remote_send_weight)
            
            classified = {
                'text2text': [],
                'image2text': [],
                'text2image': [],
                'image2image': [],
                'text2video': []
            }
            
            classified_avg_time = {
                'text2text': None,
                'image2text': None,
                'text2image': None,
                'image2image': None,
                'text2video': None 
            }

            classified_nums = {
                'text2text': None,
                'image2text': None,
                'text2image': None,
                'image2image': None,
                'text2video': None
            }
        
        
        time.sleep(5)

