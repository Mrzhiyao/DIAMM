import os
import json
import glob
from collections import defaultdict
from datetime import datetime

def process_json_files(folder_path):
    """
    处理文件夹中的所有JSON文件，按任务ID整理所有数据记录
    
    参数:
        folder_path (str): 包含JSON文件的文件夹路径
    
    返回:
        dict: 按任务ID组织的字典 {task_id: [所有记录列表]}
    """
    import os
    import glob
    import json
    from datetime import datetime
    from collections import defaultdict
    import re
    
    # 获取所有JSON文件路径
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"警告: 在 {folder_path} 中没有找到JSON文件")
        return {}
    
    # print(f"找到 {len(json_files)} 个JSON文件，开始处理...")
    
    # 用于保存按ID分类的所有任务数据记录
    tasks_by_id = defaultdict(list)
    processed_count = 0
    skipped_count = 0
    
    # 从文件名中提取时间戳 (假设文件名格式: tasks_YYYYMMDD_HHMMSS.json)
    def extract_timestamp(filename):
        match = re.search(r'tasks_(\d{8})_(\d{6})', filename)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return datetime.min
    
    # 按文件创建时间排序 (最早的在前面)
    json_files.sort(key=lambda f: extract_timestamp(os.path.basename(f)))
    
    for file_path in json_files:
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 处理JSON文件中的所有任务类别
            for category in ["text2text", "image2text", "text2image", 
                            "image2image", "text2video"]:
                
                # 检查该类别是否存在并且有数据
                if category in data and isinstance(data[category], list):
                    for task in data[category]:
                        # 验证任务数据有效性
                        if (isinstance(task, dict) and 
                            "task_id" in task and 
                            task["task_id"]):
                            
                            # 提取必要信息，增加文件时间戳
                            task_data = {
                                "task_id": task["task_id"],
                                "task_type": task.get("task_type", ""),
                                "create_time": task.get("create_time", ""),
                                "description": task.get("description", ""),
                                "status": task.get("status", ""),
                                "wait_time": task.get("wait_time", None),
                                "source_file": os.path.basename(file_path),
                                "file_timestamp": extract_timestamp(os.path.basename(file_path)),
                                "service_ip": task.get("service_ip", None),
                                "container": task.get("container", None)
                            }
                            
                            # 添加到任务ID的记录列表中 (不覆盖已有记录)
                            tasks_by_id[task["task_id"]].append(task_data)
                            processed_count += 1
                        else:
                            skipped_count += 1
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
            skipped_count += 1
    
    # print(f"处理完成! 总记录数: {processed_count}")
    return dict(tasks_by_id)  # 转换为普通字典返回


def save_results(tasks_dict, output_file="tasks_by_id.json"):
    """
    将处理结果保存为JSON文件
    
    参数:
        tasks_dict (dict): 按ID组织的任务数据
        output_file (str): 输出文件名
    """
    # 按任务创建时间排序
    sorted_tasks = sorted(tasks_dict.values(), 
                         key=lambda x: x.get("create_time", ""))
    
    # 将列表转换为目标格式
    result = {}
    for task in sorted_tasks:
        result[task["task_id"]] = task
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")

