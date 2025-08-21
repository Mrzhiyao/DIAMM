import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError
from datetime import datetime
from collections import defaultdict
import json
import ast

FLOWER_API_URL = "http://192.168.2.75:9802/api/tasks"
FLOWER_USERNAME = "admin"
FLOWER_PASSWORD = "123456"

def format_timestamp(ts):
    """将 Unix 时间戳转换为可读时间"""
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, OSError):
        return "Invalid Timestamp"

def get_non_success_tasks():
    """获取所有状态非 SUCCESS 的任务"""
    try:
        response = requests.get(
            FLOWER_API_URL,
            auth=HTTPBasicAuth(FLOWER_USERNAME, FLOWER_PASSWORD),
            timeout=10
        )
        response.raise_for_status()
        tasks = response.json()
        
        if not tasks:
            print("未找到任何任务")
            return

        # 过滤非 SUCCESS 状态的任务
        filtered_tasks = {
            tid: info for tid, info in tasks.items() 
            if info.get("state", "未知").upper() != "SUCCESS"
        }

        if not filtered_tasks:
            # print("没有非 SUCCESS 状态的任务")
            return

        # 格式化输出
        # print("{:<40} {:<12} {:<25} {:<30}".format(
        #     "任务ID", "状态", "接收时间", "结果/错误"
        # ))
        # print("-" * 105)
        
        for task_id, task_info in filtered_tasks.items():
            try:
                status = task_info.get("state", "未知")
                result = str(task_info.get("result", "")).replace('\n', ' ')[:30]
                received_ts = task_info.get("received")
                received_time = format_timestamp(received_ts)
                
                if status == "FAILURE":
                    error = task_info.get("exception", "无错误信息")
                    result = f"错误: {str(error)[:50]}"  # 截断错误信息
                
                # print("{:<40} {:<12} {:<25} {:<30}".format(
                #     task_id, status, received_time, result
                # ))
            except Exception as e:
                print(f"处理任务 {task_id} 时出错: {str(e)}")
        return filtered_tasks

    except HTTPError as e:
        print(f"HTTP 错误: {e.response.status_code} - {e.response.reason}")
        return None
    except Exception as e:
        print(f"其他错误: {e}")
        return None
    
def get_container_online_counts(celery_task):

    container_counts = defaultdict(int)
    for task_id, task_info in celery_task.items():
        args_str = task_info['args']
        try:
            args = ast.literal_eval(args_str)
            if len(args) >= 5:
                metadata = args[4]
                container = metadata.get('task_type')
                # container = metadata.get('container')
                if container:
                    container_counts[container] += 1
        except Exception as e:
            print(f"解析任务 {task_id} 的args时出错: {e}")

    return dict(container_counts)


def get_container_online_new(celery_task):

    container_counts = defaultdict(int)
    for task_id, task_info in celery_task.items():
        args_str = task_info['args']
        try:
            args = ast.literal_eval(args_str)
            if len(args) >= 5:
                metadata = args[4]
                # container = metadata.get('task_type')
                container = metadata.get('container')
                if container:
                    container_counts[container] += 1
        except Exception as e:
            print(f"解析任务 {task_id} 的args时出错: {e}")

    return dict(container_counts)


