from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Text, TIMESTAMP, Enum, UUID, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, joinedload
from pgvector.sqlalchemy import Vector  # pgvector 扩展的类型
from enum import Enum as PyEnum
from datetime import datetime
import uuid
import os
from ..models.embedding import process_text, process_image
import ast  
import re
import traceback
from sqlalchemy import select
import aiofiles
from pathlib import Path
import aiohttp
import cv2
import asyncio
import tempfile
from typing import List
import hashlib
# 全局配置
OUTPUT_BASE = Path("/yaozhi/vLLM-k8s-operator/outputs")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)  # 确保基础目录存在

# 定义任务状态的枚举
class TaskStatus(PyEnum):
    in_progress = "In Progress"
    completed = "Completed"
    failed = "Failed"

Base = declarative_base()
# 定义任务模型
class Task(Base):
    __tablename__ = 'tasks'
    __table_args__ = {'schema': 'coco'}

    task_id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    task_type = Column(String, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default='In Progress')  # 使用 String 类型
    result = Column(Text, default="Task is being processed...")
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    pod_ip = Column(String, nullable=True, default=None)  # 新增字段：存储 pod 的 IP
    file_path = Column(String, nullable=True)
    vector = Column(Vector(768))  # 使用 pgvector 存储缩略图特征向量

class TaskResult(Base):
    __tablename__ = 'task_results'
    __table_args__ = {'schema': 'coco'}

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    task_id = Column(UUID, ForeignKey('coco.tasks.task_id'), nullable=False, index=True)
    result_description = Column(Text, nullable=True)  # 处理结果描述
    result_file_path = Column(String, nullable=True)  # 处理结果的文件路径
    created_at = Column(TIMESTAMP, default=datetime.utcnow)  # 结果生成时间
    vector = Column(Vector(768))  # 使用 pgvector 存储缩略图特征向量

class TaskResultFile(Base):
    __tablename__ = 'task_results_file'
    __table_args__ = {'schema': 'coco'}

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    task_id = Column(UUID, ForeignKey('coco.tasks.task_id'), nullable=False, index=True)
    result_description = Column(Text, nullable=True)  # 文件形式
    result_file_path = Column(String, nullable=True)  # 处理结果的文件路径
    created_at = Column(TIMESTAMP, default=datetime.utcnow)  # 结果生成时间
    vector = Column(Vector(768))  # 使用 pgvector 存储缩略图特征向量

async def insert_task_record(db, task):
    
    try:
        db.add(task)
        # await db.commit()  # 异步提交
    except Exception as e:
        # await db.rollback()  # 回滚事务
        print(f"数据库操作失败: {e}")
        raise

async def convert_resultpath_to_url(path):
    # 使用正则表达式提取文件名
    filename = re.search(r'([^/]+\.png)', path)
    
    if filename:
        # 拼接新的 URL
        new_url = f'http://192.168.2.75:12365/outputs/{filename.group(1)}'
        return new_url
    else:
        return "Invalid path format"

async def download_file(url: str, save_path: str) -> bool:
    """异步下载文件到指定路径"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(save_path, 'wb') as f:
                        await f.write(await response.read())
                    return True
    except Exception as e:
        print(f"下载失败: {str(e)}")
    return False

async def is_video_url(url: str) -> bool:
    """检查URL是否包含视频文件"""
    return any(url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

async def get_video_output_dir(video_url: str) -> Path:
    """生成视频专属输出目录"""
    # 从URL提取唯一标识（示例逻辑，可根据需求调整）
    video_name = Path(video_url).stem  # 获取文件名（不含扩展名）
    sanitized_name = "".join(c if c.isalnum() else "_" for c in video_name)  # 过滤特殊字符
    unique_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]  # 添加哈希防止冲突
    
    # 创建格式：原文件名_哈希
    dir_name = f"{sanitized_name}_{unique_hash}"
    output_dir = OUTPUT_BASE / dir_name
    
    # 确保目录存在
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir

async def async_extract_frames(video_path: str, output_dir: Path, interval_sec: int = 1) -> List[str]:
    """异步抽帧并保存到指定目录"""
    loop = asyncio.get_running_loop()
    
    # 获取视频元数据
    def get_metadata():
        cap = cv2.VideoCapture(str(video_path))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_rate, total_frames
    
    frame_rate, total_frames = await loop.run_in_executor(None, get_metadata)
    duration_sec = total_frames / frame_rate
    
    # 并行处理每个时间点
    tasks = []
    for sec in range(0, int(duration_sec), interval_sec):
        task = asyncio.create_task(
            _async_process_frame(video_path, output_dir, frame_rate, sec)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return [str(p) for p in results if p]

async def _async_process_frame(video_path: str, output_dir: Path, frame_rate: float, sec: int) -> Path:
    """处理单帧到指定目录"""
    loop = asyncio.get_running_loop()
    frame_pos = int(sec * frame_rate)
    
    # 读取帧
    def read_frame():
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    
    frame = await loop.run_in_executor(None, read_frame)
    if frame is None:
        return None
    
    # 生成保存路径
    frame_path = output_dir / f"frame_{sec:04d}.jpg"
    
    # 编码并保存
    def encode_frame():
        return cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
    
    img_data = await loop.run_in_executor(None, encode_frame)
    async with aiofiles.open(frame_path, 'wb') as f:
        await f.write(img_data)
    return frame_path

async def split_long_string(input_str, max_length=20):
    """
    将长字符串分割成最大长度为 max_length 的短字符串列表
    
    Args:
        input_str: 输入字符串
        max_length: 每个子字符串的最大长度（默认为20）
    
    Returns:
        list: 分割后的字符串列表
    """
    if not isinstance(input_str, str):
        raise ValueError("输入必须是字符串")
    
    if max_length <= 0:
        raise ValueError("最大长度必须大于0")
    
    # 如果字符串长度小于等于 max_length，直接返回
    if len(input_str) <= max_length:
        return [input_str]
    
    # 分割字符串
    result = []
    for i in range(0, len(input_str), max_length):
        # 获取子字符串
        substring = input_str[i:i+max_length]
        result.append(substring)
    
    return result


async def insert_result_by_cache(db, task_id, result_file_path, result_vector, description, qag_cache):
    video_sign = False
    if result_file_path and result_file_path.endswith('.mp4'):
        video_sign = True
        file_online_url = result_file_path
        output_dir = "/yaozhi/vLLM-k8s-operator/outputs/"
        filename = os.path.basename(result_file_path)
        save_path = os.path.join(output_dir, filename)
        
        # 下载文件
        success = await download_file(file_online_url, save_path)
        # print('success', success)

        # 创建视频的输出目录
        output_dir = await get_video_output_dir(result_file_path)
        # print(f"视频的帧图像处理结果将保存至: {output_dir}")
        frame_paths = await async_extract_frames(save_path, output_dir)
        

    elif result_file_path != None:
        file_online_url = await convert_resultpath_to_url(result_file_path)
    else:
        file_online_url = None
    # 创建数据库 session
    # try:
    # 获取任务

    # task = db.query(Task).filter(Task.task_id == task_id).first()
    result = await db.execute(
        select(Task)
        .where(Task.task_id == task_id)
    )
    task = result.scalars().first()
    if not task:
        db.close()
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    else:

        try:

            embedding_content = description
            # 创建处理结果记录
            insert_vector = (await process_text(embedding_content))[0]
            task_result = TaskResult(
                task_id=task.task_id,
                result_description=description,
                result_file_path=file_online_url,
                vector=insert_vector
            )

            try:  
                db.add(task_result)  
            except Exception as e:  
                print(f"插入task_result数据时出错: {repr(e)}")  
                # db.rollback()  # 回滚事务
        except:
            # print('拆分插入text_embedding', description)
            split_texts = await split_long_string(description, 60)
            print('split_texts', split_texts)
            for i in split_texts:
                embedding_content = i
                print('content', embedding_content)
                # 创建处理结果记录
                insert_vector = (await process_text(str(embedding_content)))[0]
                task_result = TaskResult(
                    task_id=task.task_id,
                    result_description=embedding_content,
                    result_file_path=file_online_url,
                    vector=insert_vector
                )

                try:  
                    db.add(task_result)  

                except Exception as e:  
                    print(f"插入short_task_result数据时出错: {repr(e)}")  


        if result_file_path != None and video_sign == False:
            if result_file_path.startswith("I:/train2017"):  
                result_file_path = result_file_path.replace("I:/train2017", "/nfs/ai/datasets")  
            elif result_file_path.startswith("G:/train2017"):  
                result_file_path = result_file_path.replace("G:/train2017", "/nfs/ai/datasets") 


            if result_vector == None:
                # print('embedding images')
                result_vector = (await process_image(result_file_path))[0]

            if isinstance(result_vector, str):  
                result_vector = ast.literal_eval(result_vector)  # 将字符串转换为列表
            task_result_file = TaskResultFile( 
                task_id=task.task_id,
                result_description=qag_cache,
                result_file_path=result_file_path,
                vector=result_vector
            )
            # 添加结果到数据库
            try:  
                db.add(task_result_file)  
                # db.commit()  # 提交事务  
                # print("task_result_file数据成功插入到数据库。")  
            except Exception as e:  
                print(f"插入task_result_file数据时出错: {repr(e)}")  
                # db.rollback()  # 回滚事务
            # db.refresh(task_result_file)
        
        elif result_file_path != None and video_sign == True:
            n = 0 
            for path in frame_paths:
                # print('embedding frame_images')
                result_vector = (await process_image(path))[0]

                if isinstance(result_vector, str):  
                    result_vector = ast.literal_eval(result_vector)  # 将字符串转换为列表
                task_result_file = TaskResultFile( 
                    task_id=task.task_id,
                    result_description=str(n),
                    result_file_path=path,
                    vector=result_vector
                )
                n = n + 1 
                # 添加结果到数据库
                try:  
                    db.add(task_result_file)  
                    # db.commit()  # 提交事务  
                    # print("frame_result_file数据成功插入到数据库。")  
                except Exception as e:  
                    print(f"插入frame_result_file数据时出错: {repr(e)}")  
            

