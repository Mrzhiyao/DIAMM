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
from drop_function1 import get_drop1

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

def get_drop2(result, send_weight, drop_result1, get_task_result, drop_deployment2):
    drop_result2 = count_recent_tasks_all_containers(result, threshold_seconds=60)
    print('drop_result2', drop_result2)
    
    total_task = 0
    for container, task_num in drop_result2.items():
        total_task = total_task + task_num

    if total_task != 0:
        for container, task_num in drop_result2.items():
            if task_num != 0:
                continue
            if float(task_num / total_task) < 0.1:
                container_task_num = 0
                for index_task in result[container]:
                    query = get_task_result(index_task['task_id'])
                    # if query['celery_status'] != 'SUCCESS':
                    #     container_task_num = container_task_num + 1
                    if query == []:
                        # print('not completed task', query)
                        container_task_num = container_task_num + 1
                if container_task_num == 0:
                    # print('datalatest_time_diff_seconds', container_task_num)
                    drop_deployment2.append(container)
    # print('drop_deployment2', drop_deployment2)
    print('send_weight', send_weight)

    return drop_deployment2, send_weight
