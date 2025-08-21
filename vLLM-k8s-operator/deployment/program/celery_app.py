from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, TIMESTAMP, Enum, UUID, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, joinedload, relationship
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector  # pgvector 扩展的类型
from kubernetes import client, config
import uuid
import uvicorn
import yaml
import os
import time
from datetime import datetime
clip_st_time = time.time()
from enum import Enum as PyEnum
db_st_time = time.time()
from .database.insert_result import insert_task_record, insert_result_by_cache
from .database.update import update_task_status_and_pod_ip
import aiofiles
# import requests
import re
from .threading_program import text2text_task, image2text_task, text2image_task, image2image_task, text2video_task
# from .threading_program_rag_new import text2text_task, image2text_task, text2image_task, image2image_task, text2video_task
# from .threading_without import text2text_task, image2text_task, text2image_task, image2image_task, text2video_task

import asyncio
from sqlalchemy import select
from celery import Celery, signals
# from celery.contrib.asyncio import AsyncTask  # 关键异步支持
# from .models.task_search import task_embedding_search
from .models.embedding import process_text, process_image
# 异步
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import logging
from sqlalchemy import text

from .utils import AsyncTaskLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
ASYNC_DATABASE_URL = "postgresql+asyncpg://postgres:TaskingAI321@192.168.2.75:5432/coco"

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=50,
    max_overflow=150,       # 允许突发增长
    pool_recycle=300,      # 缩短回收时间至5分钟
    pool_pre_ping=True
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False
)

# 初始化数据库
# DATABASE_URL = "postgresql+psycopg2://username:password@localhost/dbname"
# DATABASE_URL = "postgresql+psycopg2://postgres:TaskingAI321@192.168.2.75:5432/coco"

# engine = create_engine(DATABASE_URL, echo=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Base = declarative_base()
Base = declarative_base()

# from typing import AsyncGenerator
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     async with AsyncSessionLocal() as session:
#         yield session

# async def init_db_async():
#     async with async_engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#     print("Tables created successfully")

class TaskStatus(PyEnum): 
    in_progress = "In Progress"
    completed = "Completed"
    failed = "Failed"

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

    # 定义与 TaskResult 的关系  
    task_results = relationship("TaskResult", back_populates="task")  
    # 定义与 TaskResultFile 的关系  
    task_results_file = relationship("TaskResultFile", back_populates="task")  

class TaskResult(Base):
    __tablename__ = 'task_results'
    __table_args__ = {'schema': 'coco'}

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    task_id = Column(UUID, ForeignKey('coco.tasks.task_id'), nullable=False, index=True)
    result_description = Column(Text, nullable=True)  # 处理结果描述
    result_file_path = Column(String, nullable=True)  # 处理结果的文件路径
    created_at = Column(TIMESTAMP, default=datetime.utcnow)  # 结果生成时间
    vector = Column(Vector(768))  # 使用 pgvector 存储缩略图特征向量

    # 定义与 Task 的关系  
    task = relationship("Task",  viewonly=True)  

class TaskResultFile(Base):
    __tablename__ = 'task_results_file'
    __table_args__ = {'schema': 'coco'}

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    task_id = Column(UUID, ForeignKey('coco.tasks.task_id'), nullable=False, index=True)
    result_description = Column(Text, nullable=True)  # 文件形式
    result_file_path = Column(String, nullable=True)  # 处理结果的文件路径
    created_at = Column(TIMESTAMP, default=datetime.utcnow)  # 结果生成时间
    vector = Column(Vector(768))  # 使用 pgvector 存储缩略图特征向量

    # 定义与 Task 的关系  
    task = relationship("Task",  viewonly=True)  
# 创建表
# Base.metadata.create_all(bind=engine)
# asyncio.run(initialize_database())

print('databse_time', time.time()-db_st_time)

# 配置 Kubernetes 客户端
config.load_kube_config()  # 加载 kubeconfig 文件

import subprocess

def delete_k8s_resources(deployment_name, service_name, namespace):
    try:
        # 删除 Deployment
        subprocess.run(
            ["kubectl", "delete", "deployment", deployment_name, "-n", namespace],
            check=True,
            text=True,
        )
        print(f"Deployment {deployment_name} deleted successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

async def convert_resultpath_to_url(path):
    # 使用正则表达式提取文件名
    filename = re.search(r'([^/]+\.png)', path)
    filename_video = re.search(r'([^/]+\.mp4)', path)
    if filename:
        # 拼接新的 URL
        new_url = f'http://192.168.2.75:12365/outputs/{filename.group(1)}'
        return new_url
    elif filename_video:
        new_url = f'http://192.168.2.75:12365/outputs/{filename_video.group(1)}'
        return new_url
    else:
        return "Invalid path format"

# 异步文件操作
async def save_upload_file(file: UploadFile, file_path: str):
    async with aiofiles.open(file_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)


from contextlib import contextmanager, asynccontextmanager

# @contextmanager
# def db_session():
#     db = SessionLocal()
#     try:
#         yield db
#         db.commit()  # 成功时提交
#     except Exception as e:
#         # db.rollback()  # 异常时回滚
#         print('again')
#         raise
#     finally:
#         db.close()  # 确保关闭

@asynccontextmanager
async def async_db_session():
    session = AsyncSessionLocal()
    logger.debug("数据库会话已创建")
    try:
        yield session
        await session.commit()
        logger.debug("事务已提交")
    except Exception as e:
        logger.error(f"事务异常: {e}")
        await session.rollback()
    finally:
        await session.execute(text("SET enable_seqscan = on;"))
        await session.close()
        logger.debug("会话已关闭")

celery = Celery('tasks', 
                broker='redis://:123456@192.168.2.75:6379/0',
                backend='redis://:123456@192.168.2.75:6379/1',  # 必须配置结果后端
                enable_utc=True,
                task_track_started=True,  # 确保跟踪任务状态
                task_serializer='json',
                result_serializer='json',
                include=['program.celery_app'],
                accept_content=['json'],
                event_serializer='json',
                # 设置时区（避免时间戳解析错误）
                timezone='UTC',
                broker_connection_retry_on_startup=True,  # 修复启动警告
                task_soft_time_limit=800,  # 超时设置
                task_time_limit=1800,
                # worker_max_memory_per_child=512000*10,
                # task_reject_on_worker_lost = True,
                worker_prefetch_multiplier=1,  # 避免预取过多任务
                broker_pool_limit=200,          # 连接池大小
                task_compression='gzip',       # 压缩大型任务
                result_extended=True,           # 记录更多元数据
                broker_transport_options={
                    'visibility_timeout': 3600,
                    'fanout_prefix': True,
                    'fanout_patterns': True,
                    'master_name': 'b410-4090d-1'
                },
                flower=True,  # 启用 Flower 集成（可选）
                flower_port=9802,  # 指定 Flower 端口（默认 5555）
                result_expires=0.5 * 24 * 3600,  # 结果保留 7 天
                worker_enable_remote_control=True,  # 必须显式启用
                worker_send_task_events=True,       # 确保事件发送
                task_send_sent_event=True           # 发送任务开始事件
                )
async_logger = AsyncTaskLogger()
@celery.task(bind=True)
def process_task_async(self, task_type: str, description: str, task_id: str, file_path: str, container_result: str):
    """实际异步逻辑，通过 asyncio.run 调用异步函数"""
    print(f"当前任务 ID: {task_id}")  # 应与 task_id 一致
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_async_task(task_type, description,task_id,file_path, container_result))
        # self.update_state(status='SUCCESS', meta=result)
        return result
    
    except Exception as e:
        print('error')
        # self.update_state(status='FAILURE', meta={'error': str(e)})
        # raise
    finally:
        loop.run_until_complete(async_engine.dispose())
        loop.close()
        asyncio.set_event_loop(None)  # 清理全局状态

async def async_convert_url_to_local_path(url: str) -> str:

    # 定义匹配模式（使用正则表达式精确匹配）
    pattern = r'^192\.168\.2\.(7[58])(:\d+)?/'
    
    # 异步执行正则替换（虽然CPU密集型操作，但保持异步接口规范）
    converted = await asyncio.to_thread(
        re.sub,
        pattern,
        '/yaozhi/vLLM-k8s-operator/',
        url,
        flags=re.IGNORECASE
    )
    
    # 统一路径分隔符（兼容Windows）
    return converted.replace('\\', '/')

async def run_async_task(task_type: str, description: str, task_id: str, file_path: str, container_result: str):
    session = None
    results_text = None
    try:
        vvector = (await process_text(description))[0]
        await async_logger.log(task_id, 'celery, embedding of task insert')
        logger.info(f"vvector: {description, vvector[0:5]}")
        async with async_db_session() as db:
            await db.execute(text("SET enable_seqscan = off;"))
            await db.commit()  # 提交设置变更
        # try:
            await async_logger.log(task_id, 'celery, task table index setting')
            # 创建任务实例
            session = db
            if len(description)>70:
                description = description[0:65]
            task = Task(
                task_id=task_id,
                task_type=task_type,
                description=description,
                status=TaskStatus.in_progress.value,
                result="Task is being processed...",
                pod_ip=None,
                file_path=file_path,  # 存储文件路径
                vector=vvector
            )
            
            # 插入任务记录
            await insert_task_record(db, task)
            await async_logger.log(task_id, 'celery, task table insert')
            results_file = None
            # 任务类型判断和处理
            # print('container_resultcontainer_result', container_result)
            if task_type == 'text2text':
                # 文生文
                results_text = await text2text_task(container_result, task_id, db, task,  process_text)
                await async_logger.log(task_id, 'celery, get text2text task result')
                # print('text2text答案是', results_text)
                results_file = None
                # db = SessionLocal()
                qag_cache = 'None'
                await insert_result_by_cache(db, task.task_id, None, None, results_text, qag_cache)
                await async_logger.log(task_id, 'celery, insert task result')
                await update_task_status_and_pod_ip(db, task.task_id, new_pod_ip='RAG Return', result_description='The task has been completed', new_status=TaskStatus.completed.value)
                await async_logger.log(task_id, 'celery, update task status')
                # 执行删除deployment.yaml
                # delete_k8s_resources("qwen15-7b-chat", "qwen15-7b-chat-service", "vllm-k8s-operator-system")
            elif task_type == 'image2text':
                # 图生文
                # print('查看图生文任务的信息',task.description, task.file_path)
                match = re.search(r'(user_tasks/[^/]+)', task.file_path)
                if match:
                    file_name = match.group(1)
                # print('image_path', "http://192.168.2.75:12365/" + file_name)
                results_text = await image2text_task(container_result, task_id, db, task, "http://192.168.2.75:12365/" + file_name, process_text)
                await async_logger.log(task_id, 'celery, get img2text task result')
                # print('image2text答案是', results_text)
                # db = SessionLocal()
                qag_cache = 'None'
                results_file = None
                result_file_path = None
                result_vector = None
                await insert_result_by_cache(db, task.task_id, None, None, results_text, qag_cache)
                await async_logger.log(task_id, 'celery, insert task result')
                await update_task_status_and_pod_ip(db, task.task_id, new_pod_ip='RAG Return', result_description='The task has been completed', new_status=TaskStatus.completed.value)
                await async_logger.log(task_id, 'celery, update task status')
            elif task_type == 'text2image':
                # 文生图
                # print('查看文生图任务的信息',task.description, task.file_path)
                if task.file_path != None:
                    match = re.search(r'(user_tasks/[^/]+)', task.file_path)
                    if match:
                        file_name = match.group(1)
                    # print('image_path', "http://192.168.2.75:12365/" + file_name)
                    img_url = "http://192.168.2.75:12365/" + file_name
                else:
                    img_url = None
                results = await text2image_task(container_result, task_id, db, task, img_url, process_text)
                await async_logger.log(task_id, 'celery, get text2timg task result')
                print('read_results_text2image', results)
                results_text = results[0]
                results_file = results[1]
                # db = SessionLocal()
                qag_cache = 'None'
                await insert_result_by_cache(db, task.task_id, results_file, None, results_text, qag_cache)
                await async_logger.log(task_id, 'celery, insert task result')
                await update_task_status_and_pod_ip(db, task.task_id, new_pod_ip='RAG Return', result_description='The task has been completed', new_status=TaskStatus.completed.value)
                await async_logger.log(task_id, 'celery, update task status')
                # 执行删除deployment.yaml
                # delete_k8s_resources("qwen2-vl-7b-instruct", "qwen2-vl-7b-instruct-service", "vllm-k8s-operator-system")
            elif task_type == 'image2image':
            # 未定义任务
                # print('查看图生图任务的信息',task.description, task.file_path)
                if task.file_path != None:
                    match = re.search(r'(user_tasks/[^/]+)', task.file_path)
                    if match:
                        file_name = match.group(1)
                    # print('image_path', "http://192.168.2.75:12365/" + file_name)
                    img_url = "http://192.168.2.75:12365/" + file_name
                else:
                    img_url = None
                results = await image2image_task(container_result, task_id, db, task, img_url, process_text)
                await async_logger.log(task_id, 'celery, get img2img task result')
                print('read_results_image2image', results)
                results_text = results[0]
                results_file = results[1]
                # db = SessionLocal()
                qag_cache = 'None'
                await insert_result_by_cache(db, task.task_id, results_file, None, results_text, qag_cache)
                await async_logger.log(task_id, 'celery, insert task result')
                await update_task_status_and_pod_ip(db, task.task_id, new_pod_ip='RAG Return', result_description='The task has been completed', new_status=TaskStatus.completed.value)
                await async_logger.log(task_id, 'celery, update task status')

            elif task_type == 'text2video':
                # 文生图
                # print('查看文生视频任务的信息',task.description, task.file_path)
                results = await text2video_task(container_result, task_id, db, task, process_text)
                await async_logger.log(task_id, 'celery, get text2video task result')
                print('read_results_text2video', results)
                results_text = results[0]
                results_file = results[1]
                # db = SessionLocal()
                qag_cache = 'None'
                await insert_result_by_cache(db, task.task_id, results_file, None, results_text, qag_cache)
                await async_logger.log(task_id, 'celery, insert task result')
                await update_task_status_and_pod_ip(db, task.task_id, new_pod_ip='RAG Return', result_description='The task has been completed', new_status=TaskStatus.completed.value)
                await async_logger.log(task_id, 'celery, update task status')
                
        task_data = {  
            "task_id": str(task.task_id),  
            "status": TaskStatus.completed.value,  
            "result": 'The task has been completed',  
            "results": [],  # 用于存放 TaskResult 的信息  
            "result_files": []  # 用于存放 TaskResultFile 的信息  
        }  
        if results_file != None:
            image_url = await convert_resultpath_to_url(results_file)

            task_data["results"].append({ 
                "result_description": results_text,  
                "result_file_path": image_url  
                # "created_at": result.created_at.isoformat()  # 转换为 ISO 格式  
            })  
            task_data["result_files"].append({  
                "result_description": None,
                "result_file_path": results_file,  
                # "created_at": result_file.created_at.isoformat()  # 转换为 ISO 格式  
            })  
        else:
            task_data["results"].append({ 
                "result_description": results_text,  
                "result_file_path": None  
                # "created_at": result.created_at.isoformat()  # 转换为 ISO 格式  
            })  

        # print('success_task_data', task_data)
        return task_data
    except Exception as e:
        logger.error(f"任务失败: {e}")
        # 整理任务处理结果
        task_data = {  
            "task_id": str(task.task_id),  
            "status": TaskStatus.failed.value,  
            "result": task.result,  
            "results": [],  # 用于存放 TaskResult 的信息  
            "result_files": []  # 用于存放 TaskResultFile 的信息  
        }  
        if results_file != None:
            image_url = await convert_resultpath_to_url(results_file)

            task_data["results"].append({ 
                "result_description": results_text,  
                "result_file_path": image_url  
                # "created_at": result.created_at.isoformat()  # 转换为 ISO 格式  
            })  
            
            task_data["result_files"].append({  
                "result_description": None,
                "result_file_path": async_convert_url_to_local_path(results_file),  
                # "created_at": result_file.created_at.isoformat()  # 转换为 ISO 格式  
            })  
        else:
            task_data["results"].append({ 
                "result_description": results_text,  
                "result_file_path": None  
                # "created_at": result.created_at.isoformat()  # 转换为 ISO 格式  
            })  
        # await async_logger.log(task_id, 'celery, celery task completion')
        return task_data
    finally:
        await async_logger.log(task_id, 'celery, celery task completion')
        # print('over')

@celery.task
async def query_result(task_id: str):
    print('query_result_task_id', task_id)
    async with async_db_session() as db:

        # 查找任务及其相关结果  
        task = (  
            db.query(Task)  
            .options(joinedload(Task.task_results), joinedload(Task.task_results_file))  # 预加载相关结果  
            .filter(Task.task_id == uuid.UUID(task_id))  
            .first()  
        )  

        # db.close()  

        if not task:  
            raise HTTPException(status_code=404, detail="Task not found")  
        
        # 准备返回的数据  
        task_data = {  
            "task_id": str(task.task_id),  
            "status": task.status.value,  
            "result": task.result,  
            "results": [],  # 用于存放 TaskResult 的信息  
            "result_files": []  # 用于存放 TaskResultFile 的信息  
        }  

        # 添加 TaskResult 的信息  (这个需要file吗？)
        for result in task.task_results:  # 假设 TaskResult 与 Task 之间的关系是多对一  
            task_data["results"].append({ 
                "result_description": result.result_description,  
                "result_file_path": result.result_file_path,  
                "created_at": result.created_at.isoformat()  # 转换为 ISO 格式  
            })  

        # 添加 TaskResultFile 的信息  
        for result_file in task.task_results_file:  # 假设 TaskResultFile 与 Task 之间的关系是多对一  
            img_url = result_file.result_file_path
            if img_url.startswith("I:/train2017"):  
                img_url = img_url.replace("I:/train2017", "/nfs/ai/datasets")  

                # 静态文件服务的URL  
                base_url = "http://192.168.2.75:12365/train2017"  
                # 从路径中提取文件名  
                file_name = img_url.split("/")[-1]  # 获取文件名，例如 "000000147328.jpg"  
                # 构建完整的 URL  
                image_url = f"{base_url}/{file_name}"  
                print(image_url)  # 输出: http://192.168.2.75:12365/train2017/000000147328.jpg  
            elif img_url.startswith("G:/train2017"):  
                img_url = img_url.replace("G:/train2017", "/nfs/ai/datasets") 
                # 静态文件服务的URL  
                base_url = "http://192.168.2.75:12365/train2017"  
                # 从路径中提取文件名  
                file_name = img_url.split("/")[-1]  # 获取文件名，例如 "000000147328.jpg"  
                # 构建完整的 URL  
                image_url = f"{base_url}/{file_name}"  
                print(image_url)  # 输出: http://192.168.2.75:12365/train2017/000000147328.jpg  

            image_url = convert_resultpath_to_url(img_url)
            task_data["result_files"].append({  
                "result_description": result_file.result_description,  
                "result_file_path": image_url,  
                "created_at": result_file.created_at.isoformat()  # 转换为 ISO 格式  
            })  

    return task_data  



@celery.task
async def query_status(task_id: str):
    async with async_db_session() as db:
        # 查找任务
        task = db.query(Task).filter(Task.task_id == uuid.UUID(task_id)).first()

        # db.close()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # 返回任务状态
        return {"task_id": str(task.task_id), "status": task.status.value}


# def analyze_task_features(task):
#     # 根据任务类型决定 Kubernetes 部署配置
#     print(f"Analyzing task features for file: {task.file_path}")
#     print(task.description, task.file_path)
#     search_results_texts, search_results_images = task_embedding_search(task.description, task.file_path)
#     if task.file_path == None:
#         print('search_result', search_results_texts[3])
#     else:
#         print('search_result', search_results_texts[3], search_results_images[3])

#     try:
#         if float(search_results_images[3][0]) > 0.9 and search_results_images != []:
#             # 通过缓存直接返回
#             db = SessionLocal()
#             description = '根据数据库内容直接返回结果如下，是否满足您的需求？'
#             qag_cache = 'None'
#             insert_result_by_cache(db, task.task_id, search_results_images[1][0], search_results_images[2][0], description, qag_cache)
#             update_task_status_and_pod_ip(db, task.task_id, new_pod_ip='RAG Return', result_description='The task has been completed', new_status='completed')
#     except:
#         print('no high similar content')
    
#     if task.task_type == 'text':
#         return {
#             "name": f"text-task-{task.task_id}",
#             "image": "your-text-processing-image",  # 替换为实际的 Docker 镜像
#             "replicas": 1
#         }
#     elif task.task_type == 'image':
#         return {
#             "name": f"image-task-{task.task_id}",
#             "image": "your-image-processing-image",
#             "replicas": 2
#         }
#     elif task.task_type == 'video':
#         return {
#             "name": f"video-task-{task.task_id}",
#             "image": "your-video-processing-image",
#             "replicas": 1
#         }
#     else:
#         raise ValueError("Unsupported task type")

def create_deployment(deployment, task_id):
    # 创建 Kubernetes 部署
    apps_v1 = client.AppsV1Api()

    # 构建 Deployment 配置
    deployment_body = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment["name"]
        },
        "spec": {
            "replicas": deployment["replicas"],
            "selector": {
                "matchLabels": {
                    "app": deployment["name"]
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": deployment["name"]
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": deployment["name"],
                            "image": deployment["image"]
                        }
                    ]
                }
            }
        }
    }

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(parent_dir, 'yaml_lists')
    os.makedirs(output_dir, exist_ok=True)
    yaml_file_path = os.path.join(output_dir, f"{deployment['name']}.yaml")
    
    with open(yaml_file_path, "w") as file:
        yaml.dump(deployment_body, file)
    print(f"Deployment YAML 文件已保存为 {yaml_file_path}")

    # 假设创建了 Deployment 并且任务开始执行
    # 模拟任务状态的更新：实际中可以通过监控 Kubernetes 集群的 pod 状态来决定任务是否完成
    # task_statuses[task_id] = "Completed"
    # task_results[task_id] = f"Task {task_id} has been completed successfully."

