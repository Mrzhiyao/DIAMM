import requests
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from requests_toolbelt.multipart.encoder import MultipartEncoder

# 配置参数
TEST_SERVER = "http://192.168.2.75:9801"
TEST_PARAMS = {
    "task_type": "text2image",  # 测试任务类型
    "description": "A sleek white motorcycle with black and gold accents is displayed prominently on a smooth grey surface in a brightly lit exhibition hall.",  # 固定问题描述
    "file_path": None,  # 如需测试文件任务，指定文件路径
    "max_retries": 3,  # 失败重试次数
    "poll_interval": 5,  # 结果轮询间隔（秒）
    "timeout": 600  # 单任务超时时间（秒）
}

# 并发控制
CONCURRENCY = 5  # 同时运行的任务数
TOTAL_TASKS = 10  # 总任务数

class TestRunner:
    def __init__(self):
        self.timings = []
        self.success_count = 0
        self.lock = threading.Lock()
        self.task_counter = 1

    def submit_task(self, params):
        """提交任务并返回task_id"""
        multipart_data = MultipartEncoder(
            fields={
                'task_id': params['task_id'],
                'task_type': params['task_type'],
                'description': params['description'],
                'file_path': params['file_path']
            }
        )
        
        for attempt in range(params['max_retries']):
            try:
                response = requests.post(
                    f"{TEST_SERVER}/solve_task/",
                    data=multipart_data,
                    headers={'Content-Type': multipart_data.content_type},
                    timeout=10
                )
                response.raise_for_status()
                return response.json()['task_id']
            except Exception as e:
                print(f"提交失败 (第{attempt+1}次重试): {str(e)}")
                if attempt == params['max_retries'] - 1:
                    raise
                time.sleep(2)

    def wait_for_result(self, task_id, params):
        """等待任务完成并返回耗时"""
        start_time = time.time()
        while time.time() - start_time < params['timeout']:
            try:
                response = requests.get(
                    f"{TEST_SERVER}/task_result/{task_id}",
                    timeout=5
                )
                data = response.json()
                print('id', task_id, data)
                if data['business_status'] == 'Completed':
                    return time.time() - start_time
                elif data['business_status'] == 'failed':
                    raise Exception("任务处理失败")
                    
                time.sleep(params['poll_interval'])
            except requests.RequestException as e:
                print(f"轮询异常: {str(e)}")
                time.sleep(params['poll_interval'])
        
        raise TimeoutError("任务处理超时")

    def run_test_cycle(self, params):
        """单个测试流程"""
        with self.lock:
            cycle_num = self.task_counter
            self.task_counter += 1
        
        # print(f"\n▶ 开始第 {cycle_num} 次测试")
        try:
            # 提交阶段计时
            submit_start = time.time()
            task_id = self.submit_task(params)
            submit_time = time.time() - submit_start
            return task_id
        except Exception as e:
            print(f"× 第 {cycle_num} 次测试发送失败: {str(e)}")
            return False



def send_tasks(send_task_list):
    CONCURRENCY = len(send_task_list)
    TOTAL_TASKS = len(send_task_list)
    print(f"""\n{'='*40}
    🚀 发送并发任务
    并发数量: {CONCURRENCY}
    总任务数: {TOTAL_TASKS}
    """)

    runner = TestRunner()
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # 提交所有任务
        # task_ids = [
        #     executor.submit(runner.run_test_cycle, send_task_list[task_index])
        #     for task_index in range(len(send_task_list))
        # ]
        task_ids = [
            executor.submit(runner.run_test_cycle, task_params)
            for task_params in send_task_list  # 直接遍历任务参数列表
        ]
    
        return [f.result() for f in task_ids]
    print('send success')
        
