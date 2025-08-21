import json
from pathlib import Path
from typing import Tuple, List, Dict
import cv2 

def read_json_file(file_path):
    """
    读取 JSON 文件的完整解决方案
    功能特性：
    - 自动处理文件路径
    - 完善的错误处理
    - 支持多种 JSON 数据结构
    - 返回解析后的 Python 对象
    """
    try:
        # 将路径转换为 Path 对象（更安全的路径处理）
        json_path = Path(file_path).resolve()

        # 检查文件是否存在
        if not json_path.exists():
            raise FileNotFoundError(f"文件不存在: {json_path}")

        # 检查文件后缀
        if json_path.suffix.lower() != '.json':
            print(f"警告: 文件 {json_path.name} 不是标准的 .json 扩展名")

        # 打开并读取文件
        with json_path.open('r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                # 更详细的 JSON 错误报告
                error_msg = f"""
                JSON 解析错误！
                文件: {json_path}
                错误位置: 第 {e.lineno} 行，第 {e.colno} 列
                错误内容: {e.msg}
                上下文: {e.doc[e.pos-20:e.pos+20]}
                """
                raise ValueError(error_msg)
            
            # 验证数据结构（可选）
            if not isinstance(data, (dict, list)):
                raise ValueError("JSON 根元素必须是字典或列表")

            return data

    except Exception as e:
        # 统一异常处理
        print(f"读取 {file_path} 失败: {str(e)}")
        return None

import os
import requests
from tqdm import tqdm  # 需要安装：pip install tqdm

import os
import requests
from tqdm import tqdm
from urllib.parse import urlparse

def download_file(url, save_dir, filename=None, chunk_size=1024, timeout=10):
    """
    下载文件并返回完整保存路径
    :param url: 要下载的文件URL
    :param save_dir: 保存目录（自动创建）
    :param filename: 可选，自定义文件名
    :return: 成功返回完整文件路径，失败返回None
    """
    try:
        # 创建保存目录（包含多层目录）
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取文件名（带扩展名）
        if not filename:
            # 从URL路径中提取文件名
            parsed_url = urlparse(url)
            url_path = parsed_url.path
            filename = os.path.basename(url_path) or "download_file"
            
            # 处理无扩展名情况
            if '.' not in filename:
                content_type = requests.head(url).headers.get('Content-Type', '')
                ext = content_type.split('/')[-1]  # 如：image/jpeg → jpeg
                filename += f".{ext}" if ext else ""

        # 构建完整保存路径
        save_path = os.path.abspath(os.path.join(save_dir, filename))
        
        # 检查文件完整性（支持断点续传）
        if os.path.exists(save_path):
            remote_size = int(requests.head(url).headers.get('Content-Length', 0))
            local_size = os.path.getsize(save_path)
            
            if remote_size == local_size:
                print(f"文件已存在: {save_path}")
                return save_path
            print(f"继续下载未完成文件: {save_path}")

        # 开始下载
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建进度条
        progress = tqdm(
            total=total_size, 
            unit='B',
            unit_scale=True,
            desc=f"下载 {filename}",
            ncols=80
        )

        # 写入文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # 过滤保持连接的空白块
                    f.write(chunk)
                    progress.update(len(chunk))
        
        progress.close()
        
        # 最终校验
        if total_size > 0 and os.path.getsize(save_path) != total_size:
            raise IOError("文件下载不完整")
            
        return os.path.abspath(save_path)  # 返回绝对路径

    except Exception as e:
        print(f"下载失败: {str(e)}")
        # 清理不完整文件
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
        return None
    





from PIL import Image
import os
import logging
from typing import Tuple, Union

def validate_bbox(bbox: list, img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    验证并转换边界框坐标为有效整数坐标
    :param bbox: 边界框信息 [x_min, y_min, width, height] 或 [x_min, y_min, x_max, y_max]
    :param img_size: 元组 (width, height) 表示图像尺寸
    :return: 有效整数坐标 (left, upper, right, lower)
    """
    try:
        # 自动检测坐标格式
        if len(bbox) != 4:
            raise ValueError("边界框需要4个数值参数")

        # 假设输入为 [x_min, y_min, width, height]
        if bbox[2] < 1 and bbox[3] < 1:  # 判断是否为归一化坐标
            x_min = bbox[0] * img_size[0]
            y_min = bbox[1] * img_size[1]
            width = bbox[2] * img_size[0]
            height = bbox[3] * img_size[1]
            x_max = x_min + width
            y_max = y_min + height
        else:  # 假设为绝对坐标 [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox

        # 转换浮点为整数
        x_min = int(round(x_min))
        y_min = int(round(y_min))
        x_max = int(round(x_max))
        y_max = int(round(y_max))

        # 坐标有效性检查
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_size[0], x_max)
        y_max = min(img_size[1], y_max)

        if x_min >= x_max or y_min >= y_max:
            raise ValueError("无效的边界框坐标")

        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        logging.error(f"边界框验证失败: {str(e)}")
        raise


def get_unique_filename(directory: str, filename: str) -> str:
    """
    生成唯一文件名，避免覆盖现有文件
    :param directory: 目标目录
    :param filename: 原始文件名
    :return: 唯一文件名
    """
    base = Path(filename).stem  # 文件名主体（不含扩展名）
    ext = Path(filename).suffix  # 文件扩展名
    
    counter = 1
    new_path = Path(directory) / filename
    
    # 自动递增序号直到文件名唯一
    while new_path.exists():
        new_filename = f"{base}_{counter}{ext}"
        new_path = Path(directory) / new_filename
        counter += 1
    
    return new_path.name  # 返回处理后的文件名

def single_crop_save(
    src_image_path: str,
    bbox: list,
    output_dir: str,
    overwrite: bool = False
) -> dict:
    """
    单边界框裁剪保存
    :param src_image_path: 原始图像路径
    :param bbox: COCO格式边界框 [x,y,w,h]
    :param output_dir: 保存目录
    :param overwrite: 是否覆盖已存在文件
    :return: 包含状态和路径的字典
    """
    # 验证输入
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {"status": "error", "message": "边界框需要4个数值参数"}

    try:
        # 读取原始图像
        if not os.path.exists(src_image_path):
            raise FileNotFoundError(f"文件不存在: {src_image_path}")
        
        image = cv2.imread(src_image_path)
        if image is None:
            raise ValueError("无法读取图像文件（可能损坏或不支持的格式）")

        # 获取图像尺寸
        img_h, img_w = image.shape[:2]

        # 转换坐标 ----------------------------------------------------------
        x, y, w, h = map(float, bbox)
        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))

        # 边界安全处理
        x = max(0, min(x, img_w - 1))          # 防止右侧越界
        y = max(0, min(y, img_h - 1))          # 防止底部越界
        w = max(0, min(w, img_w - x))          # 修正过大宽度
        h = max(0, min(h, img_h - y))          # 修正过大高度

        # 有效性验证
        if w <= 0 or h <= 0:
            return {
                "status": "error",
                "message": f"无效边界框尺寸: {w}x{h}",
                "original_size": (img_w, img_h),
                "adjusted_bbox": [x, y, w, h]
            }

        # 构建保存路径 ------------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        original_name = Path(src_image_path).name
        save_path = os.path.join(output_dir, original_name)

        # 处理文件冲突
        if os.path.exists(save_path):
            if overwrite:
                print(f"警告: 覆盖已存在文件 {save_path}")
            else:
                base = Path(original_name).stem
                ext = Path(original_name).suffix
                counter = 1
                while os.path.exists(save_path):
                    new_name = f"{base}_dup{counter}{ext}"
                    save_path = os.path.join(output_dir, new_name)
                    counter += 1

        # 执行裁剪保存
        cropped = image[y:y+h, x:x+w]
        cv2.imwrite(save_path, cropped)

        return {
            "status": "success",
            "original_size": (img_w, img_h),
            "adjusted_bbox": [x, y, w, h],
            "saved_path": save_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"运行时错误: {str(e)}"
        }
    

import os 
import datetime
import time

def write_id_time_to_file(file_path, id_time_pairs):
    """
    将 ID 和时间对写入文本文件
    
    Args:
        file_path: 文件路径
        id_time_pairs: ID 和时间对的列表，格式为 [(id, time), ...]
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 以追加模式打开文件
        with open(file_path, 'a') as f:
            for task_type, id_str, time_obj in id_time_pairs:
                print('time_obj', time_obj)
                # 格式化时间
                if isinstance(time_obj, datetime.datetime):
                    time_str = time_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 保留毫秒
                else:
                    time_str = str(time_obj)
                
                # 写入行
                f.write(f"{task_type}, {id_str}, {time_str}\n")
        
        print(f"成功写入 {len(id_time_pairs)} 条记录到 {file_path}")
        return True
    
    except Exception as e:
        print(f"写入文件失败: {str(e)}")
        return False

def write_id_time_to_file_quality(file_path, id_time_pairs):
    """
    将 ID 和时间对写入文本文件
    
    Args:
        file_path: 文件路径
        id_time_pairs: ID 和时间对的列表，格式为 [(id, time), ...]
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 以追加模式打开文件
        with open(file_path, 'a') as f:
            for id_str, time_obj in id_time_pairs:
                print('time_obj', time_obj)
                # 格式化时间
                if isinstance(time_obj, datetime.datetime):
                    time_str = time_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 保留毫秒
                else:
                    time_str = str(time_obj)
                
                # 写入行
                f.write(f"{id_str}, {time_str}\n")
        
        print(f"成功写入 {len(id_time_pairs)} 条记录到 {file_path}")
        return True
    
    except Exception as e:
        print(f"写入文件失败: {str(e)}")
        return False
    

def create_timestamped_txt(folder_path, content=None):
    """
    在指定文件夹下创建以当前时间命名的文本文件
    
    Args:
        folder_path: 目标文件夹路径
        content: 文件内容（可选）
    
    Returns:
        str: 创建的文件的完整路径
    """
    try:
        # 确保文件夹存在
        os.makedirs(folder_path, exist_ok=True)
        
        # 获取当前时间戳
        current_time = time.time()
        
        # 创建文件名（使用时间戳）
        file_name = f"timestamp_{int(current_time)}.txt"
        file_path = os.path.join(folder_path, file_name)
        
        # 创建文件并写入内容
        with open(file_path, 'w') as f:
            # 写入时间信息
            # f.write(f"Timestamp: {current_time}\n")
            # f.write(f"Formatted time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
            
            # 写入自定义内容
            if content:
                f.write("\nCustom Content:\n")
                f.write(content)
        
        print(f"文件已创建: {file_path}")
        return file_path
    
    except Exception as e:
        print(f"创建文件失败: {str(e)}")
        return None

import os
import requests
from urllib.parse import urlparse
def download_image(url, save_dir):
    """
    下载网络图片到指定目录
    
    :param url: 图片的URL地址
    :param save_dir: 保存目录路径
    :return: 保存后的完整文件路径
    """
    try:
        # 创建目标目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)
        
        # 发起HTTP GET请求
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        
        # 从URL提取文件名
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "noname_image.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # 保存文件
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        print(f"图片已成功保存至：{save_path}")
        return save_path

    except requests.exceptions.RequestException as e:
        print(f"网络请求失败：{str(e)}")
    except IOError as e:
        print(f"文件操作失败：{str(e)}")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")
    return None

def png_exists(folder_path, filename):
    """
    检查文件夹中指定PNG文件是否存在
    
    :param folder_path: 要检查的文件夹路径
    :param filename: 文件名（带或不带.png扩展名均可）
    :return: 如果文件存在且是PNG格式则返回True，否则返回False
    """
    # 确保文件名以.png结尾（忽略大小写）
    if not filename.lower().endswith('.png'):
        filename += '.png'
    
    # 构建完整文件路径
    file_path = os.path.join(folder_path, filename)
    
    # 检查文件是否存在且是普通文件（不是目录）
    return os.path.isfile(file_path)
