import requests
from prometheus_api_client import PrometheusConnect
from typing import Dict, Any
import time
# from query_gpu import query_prometheus_gpu
from dgcm_jetson import query_prometheus_gpu

CLUSTER_INITIAL_STATE = {
    "192.168.2.31": {"Name": "b410-jetson-1","Memory": 15, "GPU": 12300},
    "192.168.2.32": {"Name": "b410-jetson-2","Memory": 30, "GPU": 20600},
    "192.168.2.35": {"Name": "b410-jetson-5","Memory": 61, "GPU": 36000},
    "192.168.2.36": {"Name": "b410-jetson-6","Memory": 61, "GPU": 36000},
}
def find_ip_by_name(cluster_data, target_name):
    for ip, info in cluster_data.items():
        if info['Name'] == target_name:
            return ip
    return f"未找到名称为 {target_name} 的节点"

class ClusterMonitor:
    def __init__(self, prometheus_url: str = "http://192.168.2.75:9090"):
        self.prom = PrometheusConnect(url=prometheus_url)
        self.cluster_state = CLUSTER_INITIAL_STATE.copy()
        
    def _query_prometheus(self, query: str) -> Dict[str, float]:
        """执行 PromQL 查询并返回 {node: value} 格式数据"""
        try:
            data = self.prom.custom_query(query)
            return {
                metric["metric"]["instance"].split(":")[0]: float(metric["value"][1])
                for metric in data
            }
        except requests.exceptions.RequestException as e:
            print(f"Prometheus 连接失败: {e}")
            return {}

    # def update_gpu_usage(self):
    #     """更新 GPU 使用率（基于 DCGM_FI_DEV_GPU_UTIL）"""
    #     # 查询时按 instance 标签分组（IP:Port）
    #     gpu_values = query_prometheus_gpu()
    #     print('gpu_values', gpu_values)
    #     for node_gpu in gpu_values:
    #         # print('"Hostname', node_gpu["Hostname"])
    #         query_ip = node_gpu["Hostname"]
    #         self.cluster_state[query_ip]["GPU_Available"] = node_gpu["GPU"]

    def update_gpu_usage(self):
        """更新 GPU 使用率（基于 jtop）"""
        # 查询时按 instance 标签分组（IP:Port）
        gpu_values = query_prometheus_gpu()
        for node_gpu in gpu_values:
            # print('"Hostname', node_gpu["Hostname"])
            query_ip = find_ip_by_name(CLUSTER_INITIAL_STATE, node_gpu["Hostname"])
            self.cluster_state[query_ip]["GPU_Available"] = node_gpu["GPU"]




    def get_status(self):
        """打印当前资源状态（显示节点名称而非IP）"""
        # print("\n=== 集群资源状态更新 ===")
        # print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        new_memory_resource = 0
        new_gpu_resoure = 0
        print('self.cluster_state', self.cluster_state)
        for ip, stats in self.cluster_state.items():
            new_memory_resource = new_memory_resource + float(stats['GPU_Available'])
            new_gpu_resoure = new_memory_resource
        return new_gpu_resoure, new_memory_resource, self.cluster_state
    
    def run_monitor(self, interval: int = 60):
        """持续监控"""
        while True:
            # self.update_memory_usage()
            self.update_gpu_usage()
            new_gpu_resoure, new_memory_resource, stats = self.get_status()
            return new_gpu_resoure, new_memory_resource, stats
    
