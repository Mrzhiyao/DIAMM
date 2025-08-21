from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import uvicorn
from datetime import datetime
from enum import Enum as PyEnum
import aiofiles
import asyncio
from sqlalchemy import select
from .celery_app import process_task_async, query_status  # 正确导入方式
from pydantic import BaseModel
import logging
import json
from .utils import AsyncTaskLogger
from pathlib import Path


logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名的请求
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# 配置 Kubernetes 客户端
# config.load_kube_config()  # 加载 kubeconfig 文件
class TaskResult(BaseModel):
    task_id: str
    status: str
    result: dict | None  # 根据实际结果类型调整

class TaskStatus(BaseModel):
    task_id: str
    status: str


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

# 异步文件操作
async def save_upload_file(file: UploadFile, file_path: str):
    async with aiofiles.open(file_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

from redis.asyncio import Redis
async def get_redis():
    return Redis(
        host='192.168.2.75',
        port=6379,
        db=1,
        password='123456',
        decode_responses=False,
        max_connections=300
    )

# 示例：异步查询
async def async_get_task(task_id: str):
    redis = await get_redis()
    data = await redis.get(f"task:{task_id}")
    await redis.close()
    return json.loads(data) if data else None

async_logger = AsyncTaskLogger()

@app.post("/submit_task/")
async def submit_task(
    task_type: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(None),
):
    print(f"Received task_type: {task_type}, description: {description}, file: {file.filename if file else 'No file'}")
    
    # 生成任务 ID
    task_id = uuid.uuid4()
    print(f"[提交任务] ID: {task_id}")
    await async_logger.log(task_id, 'fastapi, task received')
    # 保存文件（如果有上传）
    file_path = None
    if file:
        file_path = f"user_tasks/{task_id}_{file.filename}"
        await save_upload_file(file, file_path)
    file_content = await file.read() if file else None  # 读取文件内容
    print(f"Received task_type: {task_type}, description: {description}, file: {file.filename if file else 'No file'}")
    # print(file)

    # 返回响应：任务已提交
    response = {"task_id": str(task_id), "status": "Task submitted successfully"}
    
    await async_logger.log(task_id, 'fastapi, task saved')

    # 写入到task_groups_online用于部署决策：
        # 生成任务ID和时间戳
    create_time = datetime.now().isoformat()
    
    # 构建任务元数据
    task_metadata = {
        "task_id": str(task_id),
        "task_type": task_type,
        "create_time": create_time,
        "description": description,
        "file": file_path,
        "status": "Pending",
        "processing_time": None
    }

    # 定义保存路径
    save_dir = Path("/yaozhi/vLLM-k8s-operator/deployment/deployment_design/task_groups_online")
    json_path = save_dir / f"{task_id}.json"
    
    try:
        # 确保目录存在
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 异步写入文件
        async with aiofiles.open(json_path, 'w') as f:
            await f.write(json.dumps(task_metadata, indent=4))
    except Exception as e:
        logger.error(f"无法保存任务元数据: {str(e)}")
        raise HTTPException(status_code=500, detail="任务元数据保存失败")

    return JSONResponse(content=response)


@app.post("/solve_task/")
async def solve_task(
    task_id: str = Form(...),
    task_type: str = Form(...),
    description: str = Form(...),
    file_path: str = Form(None),
):

    response = {"task_id": str(task_id), "status": "Task with development start"}

    # 先看需要部署服务的模型决策是否已完成
    try:
        container_result = await async_get_task(task_id)
        if container_result:
            print(f"任务容器: {container_result['container']}")
        else:
            container_result = None

        process_task_async.apply_async(
            args=(task_type, description, task_id, file_path, container_result),
            task_id=str(task_id)  # 关键：强制指定任务 ID
        )
        # print('wowa')

    except:
        print('任务没找到?redis', task_id)
    return JSONResponse(content=response)

# 查询任务状态
@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    async_result = query_status.delay(task_id)
    return {
        "task_id": task_id,
        "status": async_result.status
        }

def parse_redis_result(raw_data: bytes) -> dict:
    """
    安全解析Redis返回的任务结果数据
    返回结构始终保持一致，避免AttributeError
    """
    result_template = {
        "celery_status": "PENDING",
        "business_status": "pending",
        "results": [],
        "result_files": [],
        "error": None
    }

    if not raw_data:
        return result_template

    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as e:
        logger.error(f"Redis数据解析失败: {str(e)}")
        result_template["error"] = f"Invalid JSON: {str(e)}"
        return result_template
    except Exception as e:
        logger.error(f"未知解析错误: {str(e)}")
        return result_template

    # 确保数据结构合法
    if not isinstance(data, dict):
        result_template["error"] = "Result is not a dictionary"
        return result_template

    # 解析Celery状态
    result_template["celery_status"] = data.get("status", "PENDING")

    # 解析业务结果（深度防御）
    try:
        result = data.get("result", {})
        
        # 处理result为None的情况
        if result is None:
            result_template["business_status"] = "ResultNotReady"
            return result_template
            
        # 确保result是字典类型
        if not isinstance(result, dict):
            result_template["business_status"] = f"InvalidResultType_{type(result).__name__}"
            return result_template

        result_template.update({
            "business_status": result.get("status", "processing"),
            "results": result.get("results", []),
            "result_files": result.get("result_files", [])
        })
    except Exception as e:
        logger.error(f"业务结果解析异常: {str(e)}")
        result_template["error"] = f"Result parsing error: {str(e)}"

    return result_template

from redis.exceptions import RedisError
import redis
REDIS_POOL = redis.ConnectionPool(
    host='192.168.2.75',
    port=6379,
    db=1,
    password='123456',
    max_connections=80,
    decode_responses=False  # 保持bytes类型便于处理
)
# 查询任务结果
@app.get("/task_result/{task_id}")
async def get_task_result(task_id: str):
    # 直接查询 Celery 任务状态（无需通过 Celery worker）
    # task_result = AsyncResult(task_id)
    max_retries= 30
    retry_delay = 0.5
    response = {
        "task_id": task_id,
        "celery_status": "PENDING",
        "business_status": "pending",
        "results": [],
        "result_files": [],
        "retries": 0,
        "error": None
    }

    # 初始化Redis连接
    try:
        r = redis.Redis(connection_pool=REDIS_POOL)
    except RedisError as e:
        logger.error(f"Redis连接失败: {str(e)}")
        response["error"] = f"Redis connection error: {str(e)}"
        return response

    for attempt in range(max_retries):
        try:
            # 优先从Redis获取原始数据
            raw_data = r.get(f"celery-task-meta-{task_id}")
            # print('raw_data', raw_data)
            parsed = parse_redis_result(raw_data)
            
            # 合并响应数据
            response.update({
                "celery_status": parsed["celery_status"],
                "business_status": parsed["business_status"],
                "results": parsed["results"],
                "result_files": parsed["result_files"],
                "retries": attempt + 1,
                "error": parsed["error"]
            })

            # 最终状态判断
            if parsed["celery_status"] in ("SUCCESS", "FAILURE"):
                logger.info(f"任务 {task_id} 最终状态: {parsed['celery_status']}")
                return response

            # 指数退避重试
            await asyncio.sleep(retry_delay)


        except RedisError as e:
            logger.warning(f"Redis操作异常（尝试 {attempt+1}）: {str(e)}")
            response["error"] = f"Redis operation error: {str(e)}"
        except Exception as e:
            logger.error(f"未知查询异常: {str(e)}")
            response["error"] = f"Unexpected error: {str(e)}"
            break

    # 最终检查
    try:
        final_raw = r.get(f"celery-task-meta-{task_id}") or b'{}'
        final_parsed = parse_redis_result(final_raw)
        response.update(final_parsed)
        response["retries"] = max_retries
    except Exception as e:
        logger.error(f"最终检查失败: {str(e)}")

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.2.75", port=9801)

