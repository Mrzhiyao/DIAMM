import json
import os
import time
from glob import glob
from send_weight import scp_to_servers
from check_celery import get_non_success_tasks
nodes_ip = {
    'b410-4090d-1': '192.168.2.75', 
    'b410-4090d-2': '192.168.2.190', 
    'b410-4090d-3': '192.168.2.78', 
    'b410-3090-1': '192.168.2.80', 
    'b410-2070s-1': '192.168.2.5', 
    'b410-2070s-2': '192.168.2.6', 
    'b410-2070s-3': '192.168.2.7', 
    # 'b410-2070s-4': '192.168.2.133', 
}

remote_base_dir = "/home/yaozhi/images"


def process_json_files(directory, remove_item):
    """处理目录中的所有JSON文件，删除指定的内容项"""
    # 获取目录中的所有JSON文件
    json_files = glob(os.path.join(directory, '*.json'))
    
    for file_path in json_files:
        modified = False
        # 读取JSON文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ 文件 '{file_path}' 格式错误，跳过处理")
                    continue
            
            # 确保数据是列表的列表
            if isinstance(data, list) and all(isinstance(item, list) for item in data):
                # 过滤掉匹配项
                new_data = [
                    item for item in data 
                    if not (len(item) == 2 and [str(item[0]), str(item[1])] == remove_item)
                ]
                
                # 检查是否做了修改
                if len(new_data) != len(data):
                    modified = True
                    data = new_data
            
            # 如果数据被修改且不为空，写回文件
            if modified:
                if data:  # 如果还有数据，写入文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    print(f"✅ 已更新文件: {file_path}")
                else:  # 如果数据为空，删除文件
                    os.remove(file_path)
                    print(f"🚫 文件已清空，删除: {file_path}")
        except Exception as e:
            print(f"❌ 处理文件 '{file_path}' 时出错: {str(e)}")

directory_path = '/nfs/ai/send_weight/send_lists'

def deduplicate_and_clean(folder_path):
    """定时处理JSON文件的主函数"""
    while True:
        # 获取所有JSON文件路径
        json_files = glob(os.path.join(folder_path, "*.json"))
        
        unique_data = set()  # 使用集合去重
        processed_files = []  # 记录成功处理的文件

        # 处理每个JSON文件
        for file_path in json_files:
            try:
                # 尝试读取文件
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 将条目转换为元组存入集合去重
                for item in data:
                    unique_data.add(tuple(item))
                
                processed_files.append(file_path)
            
            except (json.JSONDecodeError, PermissionError, IOError) as e:
                print(f"跳过文件 {os.path.basename(file_path)}，原因：{str(e)}")
                continue

        for data in unique_data:
            # print(data)
            target_ips = data[0]
            local_folder = data[1]
            print(target_ips, local_folder, remote_base_dir)
            if target_ips == 'b410-2070s-4':
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if target_ips == 'b410-3090-1' and '7b' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if target_ips == 'b410-3090-1' and '7B' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if 'Qwen2-VL-7B-Instruct' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if 'Qwen2.5-7B-Instruct' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if 'glm-4-9b-chat' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if 'Meta-Llama-3.1-8B-Instruct' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if 'CogVideo-1.0' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if target_ips == 'b410-2070s-1' and 'VL' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if target_ips == 'b410-2070s-2' and 'VL' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue
            if target_ips == 'b410-2070s-3' and 'VL' in local_folder:
                remove_item = [target_ips, local_folder]
                process_json_files(directory_path, remove_item)
                continue

            # if target_ips == 'b410-3090-1' and '8b' in local_folder:
            #     continue   
            # 
            celery_task1 = get_non_success_tasks()
            time.sleep(10)
            celery_task2 = get_non_success_tasks()

            if celery_task1 == None and celery_task2 == None:   
                scp_to_servers([nodes_ip[target_ips]], local_folder, remote_base_dir)
                remove_item = [target_ips, local_folder]
                print('remove_item', remove_item)
                process_json_files(directory_path, remove_item)
                # time.sleep(10)
            else:
                continue
        
        print('权重传输需求检测')

        time.sleep(5)

if __name__ == "__main__":
    # 使用示例（需要替换实际路径）
    target_folder = "/nfs/ai/send_weight/send_lists"
    
    # 添加启动保护
    try:
        deduplicate_and_clean(target_folder)
    except KeyboardInterrupt:
        print("\n程序已安全退出")
