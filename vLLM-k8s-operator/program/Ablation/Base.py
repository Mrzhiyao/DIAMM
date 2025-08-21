import threading
import time
import subprocess
import requests
from openai import OpenAI
import webuiapi  
from PIL import Image
import psycopg2
from .embedding import get_embedding
import numpy as np
import os
import aiohttp, asyncio
import aiofiles
from sqlalchemy import text
from openai import AsyncOpenAI
import traceback
from urllib.parse import urlparse
import base64
from io import BytesIO
from .utils import AsyncTaskLogger
from typing import Optional, Dict, Any
import json
import asyncio
from .kubernets_query import find_container_info
import re
import sys

nodes_ip = {
    'b410-4090d-1': '192.168.2.75', 
    'b410-4090d-2': '192.168.2.190', 
    'b410-4090d-3': '192.168.2.78', 
    'b410-3090-1': '192.168.2.80', 
    'b410-2070s-1': '192.168.2.5', 
    'b410-2070s-2': '192.168.2.6', 
    'b410-2070s-3': '192.168.2.7', 
    'b410-2070s-4': '192.168.2.133', 
}

prompt_t2t = r"""You are an AI assistant that must obey these core protocols:

    1. ​**Language Mirroring**  
    Always respond in the exact language the question was asked

    2. ​**Token Enforcement**  
    Hard limit: 70 tokens (including spaces/punctuation)

    3. ​**Safety Mechanisms**  
    a. Pre-response token estimation  
    b. Post-generation truncation if >70 tokens  
    c. [...] substitution at 68 tokens as fail-safe

    4. ​**Output Format**  
    [Condensed Answer]   
    Example: "Quantum entanglement demonstrates [...] "

    **Execution Workflow:**  
    1. Detect input language  
    2. Draft complete answer  
    3. Compress using:  
    - Stopword removal  
    - Synonym substitution  
    - Clause simplification  
    4. Verify twice with tokenizer  
    5. Apply truncation if needed

    **Absolute Restrictions:**  
    × No markdown formatting  
    × No line breaks  
    × No apology about length

    Now respond to:  
    [USER_QUESTION]
    """

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


SAVE_DIR = "/yaozhi/vLLM-k8s-operator/deployment/deployment_design/tasks_ip"  # 可根据需求修改路径
async_logger = AsyncTaskLogger()
async def generate_context_prompt(question: str, knowledge_base: list) -> str:
    """
    将知识库内容格式化为标准提示模板
    
    :param question: 需要回答的问题
    :param knowledge_base: 知识库列表，格式为 [(id, content, score), ...]
    :return: 格式化后的提示文本
    """
    context_blocks = []
    for i, entry in enumerate(knowledge_base, 1):
        # 解析不同长度的元组
        doc_id = entry[0]
        content = entry[:-1]
        score = entry[-1]

        # 构建来源描述
        source_detail = []
        if doc_id.startswith('D'):
            source_detail.append(f"技术文档 {doc_id}")
        
        source_detail.append(f"相似度分数：{score:.4f}")

        # 生成知识块
        context_block = f"<<< KP#{i} >>>\n" 
        context_block += f"• Source: {' | '.join(source_detail)}\n"
        context_block += f"• Key Insight: {content}\n"
        context_blocks.append(context_block)

    prompt = f"""Answer the following question by leveraging both general knowledge and the specifically provided contextual information.

**Question:** {question}

**Relevant Context:**  
<<< Begin of Contextual Knowledge >>>\n"""
    prompt += '\n'.join(context_blocks)
    prompt += "<<< End of Contextual Knowledge >>>\n\n"
    prompt += """**Response Requirements:**  
1. First analyze which knowledge points are directly relevant to the question  
2. Identify any contradictions between knowledge sources  
3. Provide a synthesized answer using both contextual and general knowledge  
"""

    return prompt


async def search_vectordatabase(db, task, dbname, process_text, top_n=3):
    query_tables = """
    SELECT tablename 
    FROM pg_catalog.pg_tables 
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema', 'coco')"""
    
    # 异步执行查询
    # tables = await db.fetch(query_tables)
    result = await db.execute(text(query_tables))
    tables = result.fetchall()
    # print(f'发现 {len(tables)} 张表: {tables}')
    results_all = []
    for table in tables:
        # print('table', table[0])
        if table[0] == "edu_c_plus":
            positive_feature = await get_embedding(task)
            positive_feature = f"[{', '.join(map(str, np.array(positive_feature, dtype=np.float32).flatten()))}]"
            params = [positive_feature]  # Start with the positive feature
            # print('positive_feature', positive_feature)
            query = """
            SELECT text, start_time, end_time, features <=> :feature  AS positive_distance
            FROM edu_c_plus
            ORDER BY positive_distance LIMIT :limit;
            """
            # params.append(top_n)
            params = {"feature": positive_feature, "limit": top_n}  # 字典形式参数

        elif table[0] == "edu_c_plus_pdf":
            positive_feature = await get_embedding(task)
            positive_feature = f"[{', '.join(map(str, np.array(positive_feature, dtype=np.float32).flatten()))}]"
            # print(positive_feature)
            # params = [positive_feature]  # Start with the positive feature
            query = """
            SELECT text, page, features <=> :feature AS positive_distance
            FROM edu_c_plus_pdf
            ORDER BY positive_distance LIMIT :limit;
            """
            # params.append(top_n)
            params = {"feature": positive_feature, "limit": top_n}  # 字典形式参数

        elif table[0] == "wiki_question":
            positive_feature = await process_text(task)
            positive_feature = f"[{', '.join(map(str, np.array(positive_feature[0], dtype=np.float32).flatten()))}]"
            params = [positive_feature]  # Start with the positive feature
            query = """
            SELECT 
            q.question AS question_text,
            s.sentence AS related_sentence,
            q.features <=> :feature AS question_distance
            FROM wiki_question q
            LEFT JOIN wiki_sentence s 
            ON q.questionid = s.questionid
            WHERE s.label = '1'
            """
            query += " ORDER BY question_distance LIMIT :limit;"
            
            # params.append(top_n)
            params = {"feature": positive_feature, "limit": top_n}  # 字典形式参数
        elif table[0] == "wiki_sentence":
            positive_feature = await process_text(task)
            positive_feature = f"[{', '.join(map(str, np.array(positive_feature[0], dtype=np.float32).flatten()))}]"
            params = [positive_feature]  # Start with the positive feature
            query = """
            SELECT sentenceid, sentence, features <=> :feature AS positive_distance
            FROM wiki_sentence
            ORDER BY positive_distance LIMIT :limit;
            """

            # params.append(top_n)
            params = {"feature": positive_feature, "limit": top_n}  # 字典形式参数
        else:
            continue

        result = await db.execute(text(query), params)
        results = result.fetchall()
        results_all.append(results)

    return results_all

async def search_vector_text2img(db, task, dbname, process_text, top_n=3):
    
    try:
        # 获取文本特征
        positive_feature = await process_text(task)  # 假设process_text是异步函数
        positive_feature = f"[{', '.join(map(str, np.array(positive_feature[0], dtype=np.float32).flatten()))}]"

        results_all_text2text = []
        results_all_text2img = []

        # 文文匹配查询
        text2text_query = """
            SELECT q.fig_id AS question_text,
                   q.path AS figure_path,
                   s.description AS related_sentence,
                   s.features_desc <=> :feature AS positive_distance
            FROM image_infor q
            LEFT JOIN image_description s 
            ON q.fig_id = s.fig_id
            ORDER BY positive_distance LIMIT :limit;
        """
        text2text_params = {"feature": positive_feature, "limit": top_n}

        # 文图匹配查询
        text2img_query = """
            SELECT q.fig_id AS question_text,
                   q.path AS figure_path,
                   s.description AS related_sentence,
                   q.features <=> :feature AS positive_distance
            FROM image_infor q
            LEFT JOIN image_description s 
            ON q.fig_id = s.fig_id
            ORDER BY positive_distance LIMIT :limit;
        """
        text2img_params = {"feature": positive_feature, "limit": top_n}

        # 执行文文匹配查询
        # print('开始检索text2text')
        result = await db.execute(text(text2text_query), text2text_params)
        results = result.fetchall()
        results_all_text2text.append(results)

        # 执行文图匹配查询
        # print('开始检索text2img')
        result = await db.execute(text(text2img_query), text2img_params)
        results = result.fetchall()
        results_all_text2img.append(results)

    except Exception as e:
        # await db.rollback()
        print(f"数据库操作失败: {str(e)}")
        raise
    finally:
        # await db.close()
        return results_all_text2text, results_all_text2img


async def search_vector_text2video(db, task, process_text, top_n=3):
    
    try:
        # 获取文本特征
        positive_feature = await process_text(task)  # 假设process_text是异步函数
        positive_feature = f"[{', '.join(map(str, np.array(positive_feature[0], dtype=np.float32).flatten()))}]"

        results_all_text2img = []

        # 文图匹配查询
        text2img_query = """
            SELECT q.fig_id AS fig_id,
                q.fig_path AS figure_path,
                q.video_path AS video_path,
                q.video_text AS video_text,
                q.features <=> :feature AS positive_distance
            FROM video_youtube q
            ORDER BY positive_distance LIMIT :limit;
        """
        text2img_params = {"feature": positive_feature, "limit": top_n}

        # 执行文图匹配查询
        # print('开始检索text2img')
        result = await db.execute(text(text2img_query), text2img_params)
        results = result.fetchall()
        results_all_text2img.append(results)

    except Exception as e:
        # await db.rollback()
        print(f"数据库操作失败: {str(e)}")
        raise
    finally:
        # await db.close()
        return results_all_text2img



async def download_image(url, save_dir):
    """
    下载网络图片到指定目录
    
    :param url: 图片的URL地址
    :param save_dir: 保存目录路径
    :return: 保存后的完整文件路径
    """
    
    try:
        # 创建目标目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)
        
        # 异步HTTP请求
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url) as response:
                response.raise_for_status()  # 检查HTTP状态码
                
                # 从URL获取文件名
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or "noname_image.jpg"
                save_path = os.path.join(save_dir, filename)
                
                # 异步保存文件
                async with aiofiles.open(save_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024 * 16)  # 16KB分块读取
                        if not chunk:
                            break
                        await f.write(chunk)
                        
                # print(f"图片已成功保存至：{save_path}")
                return save_path

    except aiohttp.ClientError as e:
        print(f"网络请求失败：{str(e)}")
    except IOError as e:
        print(f"文件操作失败：{str(e)}")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")
    return None


async def convert_path(url: str) -> str:
    # 提取路径部分（支持带端口号的URL）
    path = '/'.join(url.split('/')[3:]) if '://' in url else url
    
    # 拼接新路径并标准化
    return f"/yaozhi/vLLM-k8s-operator/{path.lstrip('/')}"
async def load_image_file(img_path):
    # 强制转换为RGB模式（防止Alpha通道问题）
    img = Image.open(img_path).convert('RGB')
    
    return img

async def generate_prompt(sentence: str) -> str:
    return prompt_t2t.replace("[USER_QUESTION]", sentence)

async def execute_yaml(file_path):
    """执行 YAML 文件部署."""
    try:
        subprocess.run(["kubectl", "apply", "-f", file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while applying {file_path}: {e}")


async def check_and_get_files(path):
    # 检查路径是否存在
    if not os.path.exists(path):
        return []
    
    # 列出路径下所有条目（文件和子目录）
    all_entries = os.listdir(path)
    
    # 筛选出文件（排除子目录）
    files = [
        os.path.join(path, entry) 
        for entry in all_entries 
        if os.path.isfile(os.path.join(path, entry))
    ]
    
    return files

async def find_task_by_id(data, target_task_id):
    """
    在嵌套数据结构中查找指定task_id的任务条目
    :param data: 输入的任务数据（列表套字典结构）
    :param target_task_id: 要查找的task_id字符串
    :return: 找到的任务字典或None
    """

    for task_type, task_lists in data.items():  # 遍历字典键值对
        # print('task_lists', task_lists)
        for task in task_lists:  # 直接遍历任务列表中的每个任务字典
            if task.get('task_id') == target_task_id:
                return task
    return None

base_delay = 1  # 初始延迟基数 (秒)
max_delay = 30   # 最大延迟 (秒)
import random

async def wait_for_service(container_result, task_id, db, base_url, container_name, task, get_results, process_text, img_url, timeout=300, interval=5):
    """轮询等待服务可用."""
    # print('task', task)
    start_time = time.time()
    target_id = str(task_id)
    attempt = 0
    while time.time() - start_time < timeout:
        # 先找部署算法分配的模型
        print('container_result', time.time(), container_result, type(container_result))
        if container_result != None:
            try:
                print('try', container_result)
                out_container_name = container_result["container"]
                # 从集群里找到对应服务的信息
                find_result = await find_container_info(out_container_name)
                base_url = "http://" + nodes_ip[find_result['node']] + ":" + str(find_result['services'][0]['ports'][0]['node_port'])
                print("base_url", base_url)
                break
            except:
                print('except', container_result)

        try:
            result = await async_get_task(target_id)
            print('find_result', time.time(), result)
            if result:
                print(f"任务容器: {result['container']}")
                if result != None:
                    attempt = attempt + 1
                    try:
                        out_container_name = result["container"]
                        # 从集群里找到对应服务的信息
                        container_result = await find_container_info(out_container_name)
                        base_url = "http://" + nodes_ip[container_result['node']] + ":" + str(container_result['services'][0]['ports'][0]['node_port'])
                        print("base_url", base_url)
                        break
                    except:
                        jitter = random.uniform(0, 1)   # 生成0~1的随机数
                        delay = min(max_delay, base_delay * (2 ** attempt))  # 指数延迟
                        actual_delay = delay * jitter   # 加入抖动 ✅打破同步性！
                        time.sleep(actual_delay)

                        continue
                        # print('服务未部署好或不可用')
        except:
            # print("任务ip不完整或不可读取")
            time.sleep(0.5)
            continue

    start_time = time.time()
    try:
        if 'video' in out_container_name:
            timeout = 3000
    except:
        print('')
    while time.time() - start_time < timeout:
        if  '192.168.2.75:30011' in base_url:
            raise TimeoutError(f"找不到服务ip")
        try:
            response = await get_results(task_id, db, base_url, out_container_name, task.task_id, task.description, process_text, img_url)
            print('check_result', out_container_name, response)
            # 新增检查：如果服务是 video/stable 且 response[1] 不存在，则继续等待
            if "diffusion" in out_container_name or "video" in out_container_name:
                # 检查响应数据结构和文件实体
                print('response', response)
                if len(response) < 2 or not response[1]:
                    print(f"[{out_container_name}] 响应数据不完整，继续等待...")
                    continue
                
                # 检查文件是否真实存在
                file_path = os.path.abspath(response[1])  # 转换为绝对路径
                print('file_path', file_path, os.path.exists(file_path))
                if not os.path.exists(file_path):
                    print(f"模型未成功使用得到答案: {file_path}，延迟重试...")
                    continue
                
            return response
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            print(f"连接错误: {str(e)}")
        except Exception as e:
            print(f"未知错误: {str(e)}")
            traceback.print_exc()  # 打印完整堆栈跟踪
        await asyncio.sleep(interval)  # 替换 time.sleep
    raise TimeoutError(f"服务在 {timeout} 秒内未就绪")



async def execute_task(task_function, *args):
    """创建并启动线程以异步执行任务."""
    thread = threading.Thread(target=task_function, args=args)
    thread.start()
    return thread

async def async_chat_completion(input_text: str, model_name: str, base_url: str):
    print('t2t', base_url)
    text = await generate_prompt(input_text)
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    response = await client.chat.completions.create(
        model = model_name,
        # model="qwen15-7b",
        # model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": text  # 同步或异步函数需统一
            }
        ]
    )
    return response.choices[0].message.content

async def multimodal_generation(input_text: str, img_url: str, base_url: str):
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    response = await client.chat.completions.create(
        model="qwen2-vl-2b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]
            }
        ]
    )
    return response.choices[0].message.content

import httpx
async def text2text_model(task_id, db, base_url, container_name, id, input, process_text, img_url):
    if "ollama" in container_name:
        print("container_name", container_name)
        parts = container_name.split('-')
        pattern = re.compile(r'^\d+[bB]$')  # 匹配纯数字 + b/B 的组合
        for part in parts:
            if pattern.match(part):
                param =  part.lower()
        if param == "3b":
            model_name = "qwen2.5:3b"
        elif param == "7b":
            model_name = "qwen2.5:7b"
        elif param == "8b":
            model_name = "llama3.1:8b"
        elif param == "9b":
            model_name = "glm4:9b"
    else:
        model_name = "qwen15-7b"
    
        # 找一下参数对应
    base_url = base_url + "/v1/"
    text2text_st_time = time.time()
    
    await async_logger.log(task_id, 'process, prompt generation')
    elapsed_time = time.time()-text2text_st_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 程序耗时: {elapsed_time:.4f} 秒\n"
    # print(log_entry.strip())
    async with aiofiles.open("text2text_vector_times.txt", "a", encoding="utf-8") as f:
        await f.write(log_entry)
    
    text2text_llm_st_time = time.time()
    t2t_results = await async_chat_completion(input, model_name, base_url)
    await async_logger.log(task_id, 'process, text2text LLM')

    elapsed_llm_time = time.time()-text2text_llm_st_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LLM程序耗时: {elapsed_llm_time:.4f} 秒\n"
    # print(log_entry.strip())
    async with aiofiles.open("text2text_llm_times.txt", "a", encoding="utf-8") as f:
        await f.write(log_entry)
    return t2t_results

async def image2text_model(task_id, db, base_url, container_name, id, input, process_text, img_url):
    base_url = base_url + "/v1/"
    result = await multimodal_generation(input, img_url, base_url)
    await async_logger.log(task_id, 'process, img2text LLM')
    return result

async def async_txt2img(url: str, prompt: str, output_path: str, steps: int = 5):
    async with aiohttp.ClientSession() as session:
        try:
            # 异步发送请求
            async with session.post(
                f"{url}/sdapi/v1/txt2img",
                json={"prompt": prompt, "steps": steps, "seed": 1003, "styles": ["anime"], "cfg_scale":7},
                timeout=30
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # 添加图片数据检查
                if not data.get('images') or not data['images'][0]:
                    print("服务端返回无效的图片数据")

                # 异步保存图片
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(base64.b64decode(data['images'][0]))
                # print(f"图片已保存至: {output_path}")

        except aiohttp.ClientConnectionError as e:
            print(f"连接失败: {str(e)}")
            random_number = random.uniform(0.5, 2)
            time.sleep(random_number)

        except asyncio.exceptions.TimeoutError:
            print("请求超时")
            random_number = random.uniform(0.5, 2)
            time.sleep(random_number)

        except aiohttp.ClientResponseError as e:
            print(f"HTTP错误: {e.status} {e.message}")
            random_number = random.uniform(0.5, 2)
            time.sleep(random_number)

        except json.JSONDecodeError as e:
            print(f"响应解析失败: {str(e)}")
        except KeyError as e:
            print(f"返回数据缺少必要字段: {str(e)}")
        except OSError as e:
            print(f"文件保存失败: {str(e)}")
        except Exception as e:
            # 获取完整的异常堆栈
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f"未处理错误: {exc_type.__name__}: {str(exc_value)}")
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2)

async def async_img2img(
    url: str,
    image_path: str,
    prompt: str,
    output_path: str = "output.png",
    steps: int = 12,
    cfg_scale: float = 6.5,
    denoising_strength: float = 0.8,
    seed: int = 5555
):
    try:
        # 同步加载图片并转换为Base64（保持同步操作）
        def load_and_convert_image():
            img = Image.open(image_path).convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        image_b64 = await asyncio.to_thread(load_and_convert_image)

        # 构建有效载荷
        payload = {
            "init_images": [image_b64],
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "seed": seed
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.post(
                f"{url}/sdapi/v1/img2img",
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # 异步保存图片
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(base64.b64decode(data['images'][0]))
                
                return output_path

    except Exception as e:
        print(f"处理失败: {str(e)}")
        random_number = random.uniform(0.5, 2)
        time.sleep(random_number)
        raise


async def text2image_model(task_id, db, base_url, container_name, id, input, process_text, img_url):
    text2img_st_time = time.time()
    url_without_protocol = base_url.split("//")[1]  

    # 拆分 host 和 port  
    host, port = url_without_protocol.split(":")  
    # 创建 API 客户端，指定服务的 host 和 port  
    api = webuiapi.WebUIApi(host, port)  

    # 如果需要认证，设置用户名和密码  
    api.set_auth('username', 'password')  

    text2img_llm_st_time = time.time()
    output_path = '/yaozhi/vLLM-k8s-operator/outputs/'+ 'answer_' + str(id) + '.png'
  
    await async_txt2img(url=base_url, prompt=input, output_path=output_path,steps=30)
    await async_logger.log(task_id, 'process, text2img LLM 4')

    elapsed_llm_time = time.time()-text2img_llm_st_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] LLM程序耗时: {elapsed_llm_time:.4f} 秒\n"    
    async with aiofiles.open("text2img_llm_times.txt", "a", encoding="utf-8") as f:
        await f.write(log_entry)

    filename = output_path

    return input, filename

async def image2image_model(task_id, db, base_url, container_name, id, input, process_text, img_url):
    new_path = await convert_path(img_url)
    print('input_image', new_path)
    output_path = '/yaozhi/vLLM-k8s-operator/outputs/'+ 'answer_' + str(id) + '.png'
    # try:
    # 使用示例
    print('imag2image_base_url3:', new_path)
    # output_path = await async_img2img(
    #     url=base_url,
    #     image_path=new_path,
    #     prompt=input,
    #     output_path = output_path,
    #     steps=30
    # )
    try:
        # 调用模型生成图片
        output_path = await async_img2img(
            url=base_url,
            image_path=new_path,
            prompt=input,
            output_path=output_path,
            steps=30
        )
        
        # 验证文件是否生成
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"输出文件未生成: {output_path}")

        await async_logger.log(task_id, 'process, img2img LLM')
        return input, output_path

    except aiohttp.ClientConnectionError as e:
        print(f"服务连接失败: {str(e)}")
        raise
    except Exception as e:
        print(f"处理失败image2image: {str(e)}")
        traceback.print_exc()  # 打印完整堆栈跟踪
        raise

    return input, filename


async def async_generate_video(
    api_url: str,
    prompt: str,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    """异步调用视频生成API"""
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt}

    try:
        # 如果外部未传入session，则创建一个新的（注意需要手动关闭）
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        async with session.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            response.raise_for_status()  # 检查HTTP错误
            result = await response.json()
            return result

    except asyncio.TimeoutError:
        print("请求超时")
    except aiohttp.ClientError as e:
        print(f"请求失败: {e}")
    finally:
        if close_session and session is not None:
            await session.close()
    return None


import asyncio
import aiohttp
from urllib.parse import urlparse
from time import monotonic
from typing import Optional

async def async_check_mp4_until_available(
    url: str,
    timeout: float = 800.0,  # 总超时时间（300秒）
    check_interval: float = 5.0,  # 检测间隔（5秒）
    session: Optional[aiohttp.ClientSession] = None
) -> bool:
    """
    异步循环检测MP4文件是否可用，直到超时或成功
    
    返回:
        True: 文件在超时前可用
        False: 超时或检测失败
    """
    start_time = monotonic()
    close_session = False
    
    try:
        # 自动创建session（如果未传入）
        if not session:
            session = aiohttp.ClientSession()
            close_session = True

        while True:
            # 执行单次检测
            # print('url', url)
            available = await is_mp4_available_async(url, session=session)
            # print(available)
            if available:
                return True

            # 检查是否超时
            elapsed = monotonic() - start_time
            if elapsed >= timeout:
                return False

            # 等待下次检测
            await asyncio.sleep(check_interval)
    
    finally:
        if close_session and session:
            await session.close()

# 修改原检测函数以复用session
async def is_mp4_available_async(
    url: str,
    timeout: float = 10.0,
    session: Optional[aiohttp.ClientSession] = None
) -> bool:
    """（之前的实现，此处略作调整以复用session）"""
    """
    异步精准判断 MP4 文件是否可用
    
    参数:
        url: 要检查的视频文件URL
        timeout: 总超时时间（秒）
        session: 可复用的aiohttp会话（提升性能）
    """
    close_session = False
    try:
        # 1. 校验URL格式
        parsed = urlparse(url)
        if not all([parsed.scheme in ('http', 'https'), parsed.netloc, parsed.path]):
            return False

        # 2. 创建或复用Session
        if not session:
            session = aiohttp.ClientSession()
            close_session = True

        # 3. HEAD请求验证基础信息
        async with session.head(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
            headers={'User-Agent': 'AsyncVideoValidator/1.0'}
        ) as head_response:
            if head_response.status != 200:
                return False

            # 校验Content-Type和Content-Length
            content_type = head_response.headers.get('Content-Type', '')
            content_length = int(head_response.headers.get('Content-Length', 0))
            
            if not ('video/mp4' in content_type or url.lower().endswith('.mp4')):
                return False
            if content_length < 1024:
                return False

        # 4. 验证文件头部特征
        headers = {'Range': 'bytes=0-511'}
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as get_response:
            if get_response.status != 200 and get_response.status != 206:
                return False

            chunk = await get_response.content.read(512)
            if len(chunk) < 8:
                return False
            if chunk[4:8] not in [b'ftyp', b'moov']:  # MP4文件头特征
                return False

        return True

    except (aiohttp.ClientError, ValueError):
        return False
    finally:
        if close_session and session:
            await session.close()



async def text2video_model(task_id, db, base_url, container_name, id, input, process_text, img_url):
    api_url = base_url + '/generate_video'
    prompt = input

    result = await async_generate_video(api_url, prompt)
    result_id = result['task_id']
    filename = f'http://192.168.2.78:12365/video_outputs/output_{result_id}.mp4'

    result = await async_check_mp4_until_available(filename)
    target_directory = "/yaozhi/vLLM-k8s-operator/outputs"
    await download_image(filename, target_directory)
    return_name = filename.replace("http://192.168.2.78:12365/video_outputs", target_directory) 
    await async_logger.log(task_id, 'process, download generative video')


    return input, return_name



async def text2text_task(container_result, task_id, db, task, process_text):
    """执行文生文任务."""
    base_url = "http://192.168.2.80:31112/v1/"
    container_name = "default"
    results = await wait_for_service(container_result, task_id, db, base_url, container_name, task, text2text_model, process_text,img_url=None,)  # 等待服务启动
    return results

async def image2text_task(container_result, task_id, db, task, img_url,process_text):
    """执行图生文任务."""
    base_url = "http://192.168.2.75:31011/v1/"
    container_name = "default"
    results = await wait_for_service(container_result, task_id, db, base_url, container_name, task, image2text_model, process_text,img_url)  # 等待服务启动
    return results

async def text2image_task(container_result, task_id, db, task, img_url, process_text):
    """执行文生图任务."""
    base_url = "http://192.168.2.75:30011"
    container_name = "default"
    results = await wait_for_service(container_result, task_id, db, base_url, container_name, task, text2image_model, process_text,img_url)  # 等待服务启动
    return results

async def image2image_task(container_result, task_id, db, task, img_url, process_text):
    """执行图生图任务."""
    base_url = "http://192.168.2.75:30011"
    container_name = "default"
    results = await wait_for_service(container_result, task_id, db, base_url, container_name, task, image2image_model, process_text, img_url)  # 等待服务启动
    return results

async def text2video_task(container_result, task_id, db, task, process_text):
    """执行文生视频任务."""
    base_url = "http://192.168.2.78:31306"
    container_name = "default"
    results = await wait_for_service(container_result, task_id, db, base_url, container_name, task, text2video_model, process_text, img_url=None)  # 等待服务启动
    return results
