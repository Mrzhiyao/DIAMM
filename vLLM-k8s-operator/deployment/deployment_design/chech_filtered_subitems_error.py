def filter_duplicate_pods(data):
    """
    过滤掉重复的Pod，保留每个编号的最新实例
    参数：
        data: 包含Pod字典的列表
    
    返回：
        过滤后的列表，只保留每个Pod编号的最后出现实例
    """
    # 用于记录每个Pod编号最后出现的索引
    last_occurrence = {}
    
    # 第一步：遍历数据，记录每个Pod编号最后一次出现的索引位置
    for idx, item in enumerate(data):
        pod_name = item['pod']
        parts = pod_name.split('-')
        
        # 确保Pod名称格式正确（至少包含3个部分）
        if len(parts) < 3:
            continue
            
        pod_number = parts[1]  # 提取编号部分（如26, 35, 57等）
        last_occurrence[pod_number] = idx
    
    # 第二步：只保留每个Pod编号最后一次出现的实例
    filtered_data = []
    for pod_number, last_index in last_occurrence.items():
        filtered_data.append(data[last_index])
    
    return filtered_data

def remove_duplicate_tasks(task_list):
    # 按task_id分组
    task_groups = {}
    for task in task_list:
        task_id = task['task_id']
        if task_id not in task_groups:
            task_groups[task_id] = []
        task_groups[task_id].append(task)
    
    # 筛选要保留的任务
    filtered_tasks = []
    for task_id, tasks in task_groups.items():
        # 如果只有一个任务，直接保留
        if len(tasks) == 1:
            filtered_tasks.append(tasks[0])
        # 如果有多个任务，只保留container为None的
        else:
            none_container_tasks = [t for t in tasks if t.get('container') is None]
            # 如果有container为None的任务，保留它们
            if none_container_tasks:
                filtered_tasks.extend(none_container_tasks)
            # 如果没有None任务，保留所有（理论上不会发生）
            else:
                filtered_tasks.extend(tasks)
    
    return filtered_tasks

# 使用示例
if __name__ == "__main__":
    # 原始数据（您实际使用时替换成您的数据）
    input_data = [{'Hostname': 'b410-3090-1', 'pod': 'text2image-26-bfcbbffc9-ksv4c', 'container': 'stable-diffusion-26', 'value': 6007349248.0}, {'Hostname': 'b410-4090d-3', 'pod': 'text2image-35-76d74d8976-5kcz6', 'container': 'stable-diffusion-35', 'value': 6163509248.0}, {'Hostname': 'b410-4090d-1', 'pod': 'text2image-57-6d88f44c6b-fgh2h', 'container': 'stable-diffusion-57', 'value': 2885095424.0}, {'Hostname': 'b410-3090-1', 'pod': 'text2image-69-5cb975c975-n2ctj', 'container': 'stable-diffusion-69', 'value': 1650282496.0}, {'Hostname': 'b410-4090d-3', 'pod': 'text2image-56-64c87b7ccb-dwjcd', 'container': 'stable-diffusion-56', 'value': 832225280.0}, {'Hostname': 'b410-2070s-2', 'pod': 'text2image-15-59d9b98878-c69lc', 'container': 'stable-diffusion-15', 'value': 5578117120.0}, {'Hostname': 'b410-4090d-1', 'pod': 'text2image-80-59c85f47b6-nwvgr', 'container': 'stable-diffusion-80', 'value': 871284736.0}, {'Hostname': 'b410-3090-1', 'pod': 'text2image-81-586666596c-8rgw9', 'container': 'stable-diffusion-81', 'value': 1165529088.0}, {'Hostname': 'b410-2070s-3', 'pod': 'text2image-79-c5f549b77-mbtf4', 'container': 'stable-diffusion-79', 'value': 5345628160.0}, {'Hostname': 'b410-3090-1', 'pod': 'text2image-26-bfcbbffc9-ndqxh', 'container': 'stable-diffusion-26', 'value': 6391619584.0}, {'Hostname': 'b410-4090d-3', 'pod': 'text2image-35-76d74d8976-hv86q', 'container': 'stable-diffusion-35', 'value': 5868421120.0}, {'Hostname': 'b410-4090d-1', 'pod': 'text2image-57-6d88f44c6b-sdpbt', 'container': 'stable-diffusion-57', 'value': 1534885888.0}]
    filtered_data = filter_duplicate_pods(input_data)
