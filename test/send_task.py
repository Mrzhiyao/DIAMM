import json
import os
from collections import defaultdict
from utils import read_json_file, download_file, single_crop_save
from bingfa import send_tasks
from pathlib import Path
from datetime import datetime
import time
from utils import create_timestamped_txt, write_id_time_to_file



# 动态生成当前时间（或固定时间）
def get_formatted_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"

# 保存当前的文件
def save_task_list(task_list: list) -> str:
    # 生成文件名（格式：YYYY-MM-DD_HH-MM-SS.json）
    filename = './task_results/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    
    # 保存为 JSON 文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(task_list, f, indent=2, ensure_ascii=False)
    
    return filename

base_dir = Path('E:/com_work/datasets')
directory = './task_groups'
file_count = 0
total_stats = defaultdict(int)

# 获取目录下所有JSON文件列表
files = [f for f in os.listdir(directory) if f.endswith('.json')]
total_files = len(files)

# 初始化分类计数的字典（包含所有任务类型）
classified = {
    'text2text': 0,
    'image2text': 0,
    'text2image': 0,
    'image2image': 0,
    'text2video': 0
}
text2text_dataset = read_json_file('./source_datasets/text2text-new.json')
image2text_dataset = read_json_file('./source_datasets/image2text-new.json')
text2image_dataset = read_json_file('./source_datasets/text2image-new.json')
image2image_dataset = read_json_file('./source_datasets/image2image-new.json')
text2video_dataset = read_json_file('./source_datasets/text2video-new.json')
send_task_ids = []
receiced_task_ids = []

exceed_tasks = {
    'text2text': 1,
    'image2text': 1,
    'text2image': 1,
    'image2image': 1,
    'text2video': 1
}


def get_start_time(log_file="./task_start_times/time_records.txt"):
    """获取带毫秒的格式化时间并记录到文件"""
    now = datetime.now()
    
    # 生成精确到毫秒的时间字符串
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"
    
    # 确保目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 原子化写入操作
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Event recorded\n")
    except Exception as e:
        print(f"写入日志失败: {str(e)}")
    
    return timestamp


# 带目录路径的日志文件
print(get_start_time("./task_start_times/system_events.log"))
target_folder = r"E:\com_work\datasets\task_start_time\ab2"
# 替换占位符
current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 创建文件
created_file = create_timestamped_txt(target_folder)
if created_file:
    # 验证文件内容
    with open(created_file, 'r') as f:
        print("\n文件内容:")
        print(f.read())

for filename in files:
    file_count += 1
    file_stats = defaultdict(int)
    filepath = os.path.join(directory, filename)
    task_ids = []
    send_task_list = []
    send_type = []
    with open(filepath, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        # 统计当前文件
        for task in tasks:
            task_type = task['task_type']
            task_id =task['task_id']
            task_ids.append(task_id)
            # print(task_type, task_id)
            file_stats[task_type] += 1
            total_stats[task_type] += 1
            # print(task)
            if task_type == "text2text":
                index = classified[task_type]- int(classified[task_type]/len(text2text_dataset)) * len(text2text_dataset) 
                # print("text2text", index)
                input_description = text2text_dataset[index]["Q"]
                input_file_path = None  
                
            elif task_type == "image2text":
                input_description = "Please describe this image"
                index = classified[task_type]- int(classified[task_type]/len(image2text_dataset)) * len(image2text_dataset)
                if image2text_dataset[index]["path"].startswith("I:/train2017"):  
                    download_img_url = image2text_dataset[index]["path"].replace("I:/train2017/train2017/", "http://192.168.2.78:12365/train2017/")  
                elif image2text_dataset[index]["path"].startswith("G:/train2017"):  
                    download_img_url = image2text_dataset[index]["path"].replace("G:/train2017/train2017/", "http://192.168.2.78:12365/train2017/") 
                input_file_path = download_file(download_img_url, './image2text_img', filename=None, chunk_size=1024, timeout=10)

            elif task_type == "text2image":
                index = classified[task_type]- int(classified[task_type]/len(text2image_dataset)) * len(text2image_dataset)
                # print('text2image_index', index)
                input_description = text2image_dataset[index]["description"]
                input_file_path = None

            elif task_type == "image2image":
                index = classified[task_type]- int(classified[task_type]/len(image2image_dataset)) * len(image2image_dataset)
                # print('image2image_index', index)
                input_description = "Generate images that match the description based on the input image"
                box_info = image2image_dataset[index]["box_information"]
                if image2image_dataset[index]["path"].startswith("I:/train2017"):  
                    download_img_url = image2image_dataset[index]["path"].replace("I:/train2017/train2017/", "http://192.168.2.78:12365/train2017/")  
                elif image2image_dataset[index]["path"].startswith("G:/train2017"):  
                    download_img_url = image2image_dataset[index]["path"].replace("G:/train2017/train2017/", "http://192.168.2.78:12365/train2017/") 
                temp_file_path = download_file(download_img_url, './temp_img', filename=None, chunk_size=1024, timeout=10)

                try:
                    # 执行裁剪
                    result = single_crop_save(
                        src_image_path=temp_file_path,
                        bbox=box_info,
                        output_dir="./image2image_img",
                        overwrite=False
                    )   

                    original_path = Path(result['saved_path'])
                    print(original_path)
                    # 解析原始路径的相对部分
                    relative_path = original_path.relative_to('.')
                    input_file_path = str(base_dir / relative_path)

                except Exception as e:
                    print(f"处理失败: {str(e)}")

            elif task_type == "text2video":
                index = classified[task_type]- int(classified[task_type]/len(text2video_dataset)) * len(text2video_dataset)
                # print('text2video_index', index)
                input_description = text2video_dataset[index]["caption"]
                input_file_path = None

            print('input_file_path', input_file_path)
            TASK_PROMPT = {
                "task_type": task_type,  # 测试任务类型
                "description": input_description,
                "file_path": input_file_path,  # 如需测试文件任务，指定文件路径
                "max_retries": 3,  # 失败重试次数
                "poll_interval": 5,  # 结果轮询间隔（秒）
                "timeout": 600,  # 单任务超时时间（秒）
                "task_index": classified[task_type]
            }

            classified[task_type] = classified[task_type] + 1
            # print(TASK_PROMPT)
            send_task_list.append(TASK_PROMPT)

            send_type.append(task_type)
        print(send_task_list, 'send_task_list_old')

        receive_task_ids = send_tasks(send_task_list)

        # 只发送某一批次的任务
        # if file_count == 3:
        #     receive_task_ids = send_tasks(send_task_list)
        # else:
        #     receive_task_ids= ['1']
        #     print(file_count)
        for sub_i in range(len(receive_task_ids)): 
            id_time_pairs = [(send_type[sub_i], receive_task_ids[sub_i], datetime.now().strftime('%Y-%m-%d %H:%M:%S'))]
            # print('id_time_pairs', id_time_pairs)
            write_id_time_to_file(created_file, id_time_pairs)

        receiced_task_ids.append(receive_task_ids)

        send_type = []

    send_task_ids.append(task_ids)

    print('task_num', len(send_task_ids[-1]))
    print(receiced_task_ids[0])



    for task, task_id in zip(send_task_list, receiced_task_ids[-1]):
        task["task_id"] = task_id
        task["created_time"] = get_formatted_time()

    # print(send_task_list, 'send_task_list_new')
    saved_file = save_task_list(send_task_list)

    if file_count > 20:
        break
    time.sleep(120)
    # break


# print(classified)

   
