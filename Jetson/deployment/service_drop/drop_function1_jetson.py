from collections import defaultdict
import os
import json
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Any
from query_pods import query_pods_info
from query_gpu import query_prometheus_gpu

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


def get_drop1(drop_deployment1, result, get_task_result):
    # 过去30s未处理任何任务的模型container推出
    pods = query_pods_info()

    # 删掉没用的空闲容器
    containers = [pod['container'] for pod in pods]
    # print('all_containers', containers)

    drop_result1 = analyze_container_tasks(result)
    # 有记录的container
    container_names = list(drop_result1.keys())
    # print('container_names', container_names)

    for sub_container in containers:
        print('任务1', sub_container)
        if sub_container not in container_names and sub_container != 'redis':
            drop_deployment1.append(sub_container)


    # 打印统计结果
    for container, data in drop_result1.items():
        print('con', container)
        if data['latest_time_diff_seconds'] >= 300:
            if 'video' in container:
                if data['latest_time_diff_seconds'] >= 4000:
                    drop_deployment1.append(container)
            else:
                drop_deployment1.append(container)

        elif 60 < data['latest_time_diff_seconds'] < 300:
            print('datalatest_time_diff_seconds', data['latest_time_diff_seconds'] )
            container_task_num = 0
            for index_task in result[container]:
                query = get_task_result(index_task['task_id'])
                if query == []:
                    # print('not completed task', query) 
                    container_task_num = container_task_num + 1
            if container_task_num == 0:
                drop_deployment1.append(container)

    print('drop_deployment1', drop_deployment1)

    return drop_deployment1, drop_result1
