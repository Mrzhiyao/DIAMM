from collections import defaultdict
import os
import json
import re
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Any
from query_pods import query_pods_info
from query_gpu import query_prometheus_gpu
from drop_function1 import get_drop1
from drop_function2 import get_drop2
from drop_function3 import get_drop3
from drop_weight import get_drop_weight
from check_node_weight import check_remote_dir_exists, check_remote_file_exists
import copy
from delete_resource import delete_by_container_names
from send_weight_sign import merge_and_transfer_lists
from delete_weights import delete_remote_folder
import psycopg2
from search_result import search_result
from check_celery import get_non_success_tasks

SAVE_DIR = '../deployment_design/tasks_ip'

import requests

# 创建同步数据库连接
def create_db_connection():
    """创建 PostgreSQL 数据库连接"""
    return psycopg2.connect(
        dbname="coco",
        user="postgres",
        password="TaskingAI321",
        host="192.168.2.75",
        port="5432"
    )


# def get_task_result(task_id: str) -> dict:
#     url = f"http://mm-backend.bnuai.com/task_result/{task_id}"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # 自动检查 HTTP 错误状态（4xx/5xx）
#         return response.json()  # 直接解析 JSON 响应
#     except requests.exceptions.RequestException as e:
#         print(f"请求失败: {str(e)}")
#         return None
    
def get_task_result(task_id: str):
    try:
        conn = create_db_connection()
        # 执行查询
        results = search_result(
            db_conn=conn,
            select_id=task_id
        )
        conn.close()
        return results
    except:
        print(f"数据库查询失败")
        return None


def delete_deployment(deployment_name, namespace="vllm-k8s-operator-system"):
    """
    删除指定命名空间的 Kubernetes Deployment
    :param deployment_name: 要删除的Deployment名称
    :param namespace: 命名空间（默认default）
    :return: 删除操作状态
    """
    try:
        # 加载Kubernetes配置（自动检测环境）
        config.load_kube_config()  # 本地开发使用kubeconfig
        # config.load_incluster_config()  # 集群内Pod运行时使用
        
        # 初始化API客户端
        api = client.AppsV1Api()
        
        # 执行删除操作
        api_response = api.delete_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            # body=client.V1DeleteOptions(
            #     propagation_policy='Foreground',  # 级联删除关联Pod
            #     grace_period_seconds=5
            # )
            body=client.V1DeleteOptions(
                propagation_policy='Background',  # 先删Deployment，后台清理Pod
                grace_period_seconds=0           # 强制立即终止（慎用！）
            )
        )
        print(f"Deployment {deployment_name} 删除成功")
        return api_response
        
    except ApiException as e:
        if e.status == 404:
            print(f"Deployment {deployment_name} 不存在")
        else:
            print(f"删除失败: {e.reason}")
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
            container = task["container"]  # 新增字段
            frame = task["frame"]           # 新增字段
            key = (service_ip, container, frame)
            
            stats[task_type][key]["total"] += 1
            
            result = get_task_result(task["task_id"])
            # print(result)
            if result["celery_status"] == "SUCCESS":
                stats[task_type][key]["non_pending"] += 1

            # if task["status"] != "Pending":
            #     stats[task_type][key]["non_pending"] += 1

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
    
    # # 示例调用
    # 

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

def group_and_sort_tasks_by_container(raw_data):
    """
    将原始任务数据按 container 分组，并按 create_time 排序
    
    :param raw_data: 原始嵌套数据结构（包含文件名作为外层键）
    :return: 按 container 分组的字典，结构为 {container_name: [sorted_tasks]}
    """
    # 1. 初始化分组容器
    grouped = defaultdict(list)
    
    # 2. 遍历所有文件名下的数据（忽略文件名本身）
    for file_data in raw_data.values():
        # 3. 遍历每个任务类型（如 text2text、image2text）
        for task_type, tasks in file_data.items():
            # 跳过空任务类型（如 text2video）
            if not tasks:
                continue
            # 4. 遍历每个任务并分组
            for task in tasks:
                container = task.get("container")
                if container:
                    grouped[container].append(task)
    
    # 5. 对每个 container 的任务按 create_time 排序
    for container, tasks in grouped.items():
        # 按时间字符串升序排列（早 -> 晚）
        tasks.sort(key=lambda x: x["create_time"])
    
    return dict(grouped)

def analyze_container_tasks(grouped_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    统计每个容器的任务占比和时间差值
    :param grouped_data: 已按容器分组的任务数据，格式为 {container_name: [task_list]}
    :return: 统计结果字典，格式：
        {
            "container_name": {
                "task_count": 10,
                "latest_time_diff_seconds": 3600.5,
                "task_percentage": 25.0
            },
            ...
        }
    """
    # 计算总任务数
    total_tasks = sum(len(tasks) for tasks in grouped_data.values())
    
    stats = {}
    
    for container, tasks in grouped_data.items():
        if not tasks:
            continue
        
        # 按 create_time 排序（确保最后一个是最新任务）
        sorted_tasks = sorted(tasks, key=lambda x: x["create_time"])
        latest_task = sorted_tasks[-1]
        
        # 解析时间（兼容带时区和不带时区的 ISO 格式）
        create_time_str = latest_task["create_time"]
        # if create_time_str.endswith("Z"):
        #     create_time_str = create_time_str[:-1] + "+00:00"
        # create_time = datetime.fromisoformat(create_time_str)
        # if create_time.tzinfo is None:
        #     create_time = create_time.replace(tzinfo=timezone.utc)
        

        # 统一解析为无时区本地时间
        try:
            # 尝试解析用户定义的格式 "%Y%m%d_%H%M%S"
            create_time = datetime.strptime(create_time_str, "%Y%m%d_%H%M%S")
        except ValueError:
            # 处理ISO格式时间（强制转换为无时区）
            create_time = datetime.fromisoformat(create_time_str.replace("Z", ""))
            if create_time.tzinfo is not None:
                # 如果包含时区信息则强制转换为本地无时区
                create_time = create_time.astimezone().replace(tzinfo=None)


        # 计算时间差（UTC 时间）
        # now = datetime.now(timezone.utc)
        now = datetime.now()
        time_diff = (now - create_time).total_seconds()
        
        # 计算任务占比
        task_count = len(tasks)
        percentage = (task_count / total_tasks * 100) if total_tasks > 0 else 0
        
        stats[container] = {
            "task_count": task_count,
            "latest_time_diff_seconds": round(time_diff, 2),
            "task_percentage": round(percentage, 2)
        }
    
    return stats

# 统计规定时间内容器完成任务的情况
def count_recent_tasks_all_containers(
    grouped_data: Dict[str, List[Dict]], 
    threshold_seconds: int = 30
) -> Dict[str, int]:
    """
    统计所有容器中与当前时间差小于阈值（默认30秒）的任务数量
    :param grouped_data: 已按容器分组的任务数据
    :param threshold_seconds: 时间差阈值（秒）
    :return: 字典格式 {容器名称: 符合条件任务数量}
    """
    now = datetime.now()
    stats = {}

    for container, tasks in grouped_data.items():
        count = 0
        for task in tasks:
            # 解析时间（兼容两种格式）
            try:
                # 尝试用户自定义格式 "%Y%m%d_%H%M%S"
                create_time = datetime.strptime(task["create_time"], "%Y%m%d_%H%M%S")
            except ValueError:
                # 处理 ISO 格式时间（强制无时区）
                time_str = task["create_time"].replace("Z", "")
                create_time = datetime.fromisoformat(time_str)
                if create_time.tzinfo is not None:
                    create_time = create_time.astimezone().replace(tzinfo=None)

            # 计算时间差
            time_diff = (now - create_time).total_seconds()
            if abs(time_diff) < threshold_seconds:
                count += 1
        
        stats[container] = count
    
    return stats


def compare_dicts_for_value_one(dict1, dict2):
    # time.sleep(60)
    result = {}
    
    # 遍历 dict1 的所有 IP
    for ip in dict1:
        # 如果 dict2 中没有该 IP，跳过
        if ip not in dict2:
            continue
        
        ip_result = {}
        # 遍历 IP 下的服务类型（如 text2text）
        for service in dict1[ip]:
            # 如果 dict2 的当前 IP 下没有该服务类型，跳过
            if service not in dict2[ip]:
                continue
            
            service_result = {}
            # 遍历服务类型下的模型路径
            for model_path in dict1[ip][service]:
                # 如果 dict2 的当前服务类型下没有该模型路径，跳过
                if model_path not in dict2[ip][service]:
                    continue
                
                # 检查两个字典中该路径的值是否均为 1
                value1 = dict1[ip][service][model_path]
                value2 = dict2[ip][service][model_path]
                if value1 == 1 and value2 == 1:
                    service_result[model_path] = 1
            
            # 记录非空的服务类型结果
            if service_result:
                ip_result[service] = service_result
        
        # 记录非空的 IP 结果
        if ip_result:
            result[ip] = ip_result
    
    return result

def sanitize_container_name(model_name: str) -> str:
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
    max_base_length = 30  # 保留后缀空间
    final_name = f"{cleaned[:max_base_length]}"
    
    # 最终再次清理
    final_name = re.sub(r'[^a-z0-9-]', '', final_name)[:63]
    return final_name



tasks_type = ['text2text', 'image2text', 'text/image2image', 'text2video','text2image','image2image']

nodes_ip = {
    'b410-4090d-1': '192.168.2.75', 
    'b410-4090d-2': '192.168.2.190', 
    'b410-4090d-3': '192.168.2.78', 
    'b410-3090-1': '192.168.2.80', 
    'b410-2070s-1': '192.168.2.5', 
    'b410-2070s-2': '192.168.2.6', 
    'b410-2070s-3': '192.168.2.7', 
    'b410-2070s-4': '192.168.2.133', 
}

ips_pre = {
    '192.168.2.75': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    },
    '192.168.2.190': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    },
    '192.168.2.78': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    },
    '192.168.2.80': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    },
    '192.168.2.5': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    },
    '192.168.2.6': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    },
    '192.168.2.7': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    },
    '192.168.2.133': {
        'text2text':{'/home/yaozhi/images/Qwen2.5-3B-Instruct': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-Instruct': 0,
                     '/home/yaozhi/images/Meta-Llama-3.1-8B-Instruct': 0,
                     '/home/yaozhi/images/glm-4-9b-chat': 0,
                     '/home/yaozhi/images/Qwen2.5-3B-ollama': 0,
                     '/home/yaozhi/images/Qwen2.5-7B-ollama': 0,
                     '/home/yaozhi/images/llama31-8b-ollama': 0,
                     '/home/yaozhi/images/glm4-9b-ollama': 0
                     },
        'image2text':{'/home/yaozhi/images/Qwen2-VL-2B-Instruct': 0, 
                      '/home/yaozhi/images/Qwen2-VL-7B-Instruct': 0
                      },
        'text2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text/image2image':{'/home/yaozhi/images/yz_diffusion': 0},
        'text2video':{'/home/yaozhi/images/CogVideo-1.0': 0}
    }
}


from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import time

# 初始化调度器
scheduler = BackgroundScheduler()

ips_next = copy.deepcopy(ips_pre)
# ips_next = ips_pre

# text2text_params = [3000,7000,8000,9000]
text2text_params = [6000+1000,14000+1000,16000+1000,18000+1000]
image2text_params = [6000,16000]

text2image_params = [4000]
image2image_params = [4000]
text2video_params = [19000]

services_length = {
    'text2text': 0, 
    'image2text': 0, 
    'text2image': 0, 
    'image2image': 0, 
    'text2video': 0
}

drop_deployment1 = []
drop_deployment2 = []
drop_weight = []
send_weight = []
send_weight2 = []

ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
# # 1. 过去30s未处理任何任务的模型container推出
# drop_deployment1, drop_result1 = get_drop1(drop_deployment1, result)

# # 2. 过去30s内处理任务比例小于20%的模型container推出
# drop_deployment2, send_weight = get_drop2(result, send_weight, drop_result1, get_task_result, drop_deployment2)

# # 3. 提前发送缺失服务的模型权重到节点（参数大的，往4090上传，参数小的往2070上传）
# send_weight2 = get_drop3(services_length, send_weight2)

# print('send_weight2', send_weight2)

# # 4. 获取推出权重的列表
# mount_type = get_drop_weight(tasks_type)
# print('mount_type', mount_type)

def job_group_123():
    print('sss')
    ips_pre = copy.deepcopy(ips_next)
    # celery_task1 = get_non_success_tasks()
    # time.sleep(5)
    # celery_task2 = get_non_success_tasks()
    if 1 > 0: 
        print('deploy1')
        """每30秒执行前三个任务"""
        drop_deployment1 = []
        drop_deployment2 = []
        send_weight = []
        send_weight2 = []
        send_weight2_new = []
        all_data = load_all_classified()
        # print('all_data', all_data)
        # if all_data:
        #     print(f"共加载 {len(all_data)} 个文件")
        result = group_and_sort_tasks_by_container(all_data)
        # print('group_and_sort_tasks_by_container', result)

        # 任务1:过去30s未处理任何任务的模型container推出
        drop_deployment1, drop_result1 = get_drop1(drop_deployment1, result, get_task_result)
        
        # 任务2:过去30s内处理任务比例小于20%的模型container推出
        drop_deployment2, send_weight = get_drop2(result, send_weight, drop_result1, get_task_result, drop_deployment2)
        
        print('任务1', drop_deployment1)
        print('任务2', drop_deployment2)
        print('任务2-权重部分', send_weight)
        print('任务3', send_weight2, send_weight2_new)

        # 操作容器驱逐部分：
        if len(drop_deployment1) != 0:
            # for sub in drop_deployment1:
            delete_by_container_names(drop_deployment1)
            # print('第一步删除容器:', drop_deployment1)
        if len(drop_deployment2) != 0:
            # for sub in drop_deployment2:
            delete_by_container_names(drop_deployment2)
            # print('第二步删除容器:', drop_deployment2)
    

# def job_4():
#     """每分钟执行第四个任务"""
#     global ips_pre, ips_next, nfs_drop_local_model_weight_list, local_drop_local_model_weight_list  # 声明所有需要修改的全局变量
#     # 任务4:获取推出权重的列表
#     nfs_drop_local_model_weight_list, local_drop_local_model_weight_list = get_drop_weight(tasks_type)
    
#     # 查看本地权重的使用情况
#     pairs = []
#     for ssub in local_drop_local_model_weight_list:
#         pairs.append([nodes_ip[ssub['Hostname']], re.sub(r'-\d+$', '', ssub['container'])])

#     for ip in ips_pre:
#         # print(ips_pre[ip])
#         for task_type in ips_pre[ip]:
#             exists = any(task_type in item['pod'] for item in nfs_drop_local_model_weight_list)

#             exists_local = any(task_type in item['pod'] for item in local_drop_local_model_weight_list)
#             if exists == True:
#                 for sub_path in ips_pre[ip][task_type]: 
#                     # print(sub_path)
#                     remote_weight_path = sub_path
#                     remote_flag_path =  sub_path + '_transfer_complete.flag'
#                     r_weight = check_remote_dir_exists(ip, remote_weight_path, ssh_key_path)
#                     r_flag = check_remote_file_exists(ip, remote_flag_path, ssh_key_path)

#                     if r_weight == True and r_flag == True:
#                         model_name = os.path.basename(sub_path.rstrip('/'))
#                         if model_name == 'yz_diffusion':
#                             model_name = 'stable-diffusion'
#                         elif model_name == 'CogVideo-1.0':
#                             model_name == 'CogVideoX-2b-sat'
#                         model_name = sanitize_container_name(model_name)

#                         if [ip, model_name] in pairs:
#                             # 保留在用的权重
#                             ips_next[ip][task_type][sub_path] = 0
#                         else:
#                             ips_next[ip][task_type][sub_path] = 1
#                     else:
#                         ips_next[ip][task_type][sub_path] = 0

#             elif exists != True and exists_local == True:
#                 # 该模型除本地权重挂载在用外可以清除
#                     for sub_path in ips_pre[ip][task_type]: 
#                         # print(sub_path)
#                         remote_weight_path = sub_path
#                         remote_flag_path =  sub_path + '_transfer_complete.flag'
#                         r_weight = check_remote_dir_exists(ip, remote_weight_path, ssh_key_path)
#                         r_flag = check_remote_file_exists(ip, remote_flag_path, ssh_key_path)

#                         if r_weight == True and r_flag == True:
#                             model_name = os.path.basename(sub_path.rstrip('/'))
#                             if model_name == 'yz_diffusion':
#                                 model_name = 'stable-diffusion'
#                             elif "CogVideo" in model_name or "cogvideo" in model_name :
#                                 model_name == 'CogVideoX-2b-sat'
#                             model_name = sanitize_container_name(model_name)
#                             if 'cog' in model_name:
#                                 model_name = 'cogvideox-2b-sat'
#                             if [ip, model_name] in pairs:
#                                 # 保留在用的权重
#                                 ips_next[ip][task_type][sub_path] = 0
#                             else:
#                                 ips_next[ip][task_type][sub_path] = 1
#                         else:
#                             ips_next[ip][task_type][sub_path] = 0
#             else:
#                 for sub_path in ips_pre[ip][task_type]: 
#                     ips_next[ip][task_type][sub_path] = 0
    
#     # print('ips_next', ips_next)
#     # 复制前先对比一下ips前后的变化
#     # 执行对比
#     common_entries = compare_dicts_for_value_one(ips_pre, ips_next)
#     print("相同且值为1的条目:", common_entries)
#     import json
#     # print(json.dumps(common_entries, indent=4))

#     ips_pre = copy.deepcopy(ips_next)
            #    celery_task = get_non_success_tasks()
                # if celery_task == None: 
def job_4():
    global ips_pre, ips_next, nfs_drop_local_model_weight_list, local_drop_local_model_weight_list  # 声明所有需要修改的全局变量
    ips_pre = copy.deepcopy(ips_next)
    celery_task1 = get_non_success_tasks()
    time.sleep(20)
    celery_task2 = get_non_success_tasks()
    if celery_task1 == None and celery_task2 == None: 
        print('deploy2')
        """每分钟执行第四个任务"""
        # 任务4:获取权重列表，这些权重是现在都有的
        nfs_drop_local_model_weight_list, local_drop_local_model_weight_list = get_drop_weight(tasks_type)

        print('get_drop_weight', nfs_drop_local_model_weight_list, local_drop_local_model_weight_list)
        # 查看本地权重的使用情况
        pairs = []
        for ssub in local_drop_local_model_weight_list:
            pairs.append([nodes_ip[ssub['Hostname']], re.sub(r'-\d+$', '', ssub['container'])])

        # print('pairs', pairs)
        for ip in ips_pre:
            # print(ips_pre[ip])
            for task_type in ips_pre[ip]:
                # 该模型除本地权重挂载在用外可以清除
                for sub_path in ips_pre[ip][task_type]: 
                    # print(sub_path)
                    remote_weight_path = sub_path
                    remote_flag_path =  sub_path + '_transfer_complete.flag'
                    # print('remote_weight_pathremote_weight_path', remote_weight_path)
                    r_weight = check_remote_dir_exists(ip, remote_weight_path, ssh_key_path)
                    r_flag = check_remote_file_exists(ip, remote_flag_path, ssh_key_path)
                    if r_weight == True:
                        
                        model_name = os.path.basename(sub_path.rstrip('/'))
                        if model_name == 'yz_diffusion':
                            model_name = 'stable-diffusion'
                        elif "CogVideo" in model_name or "cogvideo" in model_name :
                            model_name == 'CogVideoX-2b-sat'
                        model_name = sanitize_container_name(model_name)
                        if 'cog' in model_name:
                            model_name = 'cogvideox-2b-sat'
                        # print('[ip, model_name]', [ip, model_name])
                        if [ip, model_name] in pairs:
                            # 保留在用的权重
                            ips_next[ip][task_type][sub_path] = 0
                        else:
                            ips_next[ip][task_type][sub_path] = 1
                            # print('?')
                    else:
                        ips_next[ip][task_type][sub_path] = 0
        
        print('ips_next', ips_next)
        # 复制前先对比一下ips前后的变化
        # 执行对比
        common_entries = compare_dicts_for_value_one(ips_pre, ips_next)
        print("相同且值为1的条目:", common_entries)
        for ip, tasks in common_entries.items():
            # 获取所有需要删除的路径（去重）
            unique_paths = set()
            for task_type, paths in tasks.items():
                for path in paths.keys():
                    unique_paths.add(path)
            
            # 删除每个唯一路径
            for path in unique_paths:
                delete_remote_folder(ip, path)
        import json
        # print(json.dumps(common_entries, indent=4))

        ips_pre = copy.deepcopy(ips_next)

# 添加定时任务
scheduler.add_job(
    job_group_123,
    trigger=IntervalTrigger(seconds=60),  # 每30秒
    id='group123_job'
)


scheduler.add_job(
    job_4,
    trigger=IntervalTrigger(minutes=5),  # 每分钟
    id='job4'
)

# 启动调度器
scheduler.start()

# 保持主线程运行
try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()






# # 查看任务完成比例
# result = calculate_non_pending_percentage(all_data)
# print('查看任务完成情况', result)

# # redis查询结果
# low_percent = find_low_percentage_entries(result, 15)
# # 存在total_tasks占比小于总total_tasks30-40%, 这个模型用完就同时删除这部分权重
# for category, info in low_percent.items():
#     print(f"\n分类 {category.upper()}")
#     print(f"总任务数: {info['total_tasks']}")
#     print("符合条件条目:")
#     for entry in info['matched_entries']:
#         print(f"  - IP: {entry['service_ip']} | 容器: {entry['container']} | 任务数: {entry['total_tasks']} | 占比: {entry['percentage']}%")    

