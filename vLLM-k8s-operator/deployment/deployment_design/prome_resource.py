import requests
from prometheus_api_client import PrometheusConnect
from typing import Dict, Any
import time
from query_gpu import query_prometheus_gpu


# 集群初始状态配置（根据您提供的参数）
CLUSTER_INITIAL_STATE = {
    "192.168.2.75": {"Name": "b410-4090d-1","Memory": 64, "GPU": 24000},
    "192.168.2.190": {"Name": "b410-4090d-2","Memory": 64, "GPU": 24000},
    "192.168.2.78": {"Name": "b410-4090d-3","Memory": 64, "GPU": 24000},
    "192.168.2.80": {"Name": "b410-3090-1","Memory": 32, "GPU": 24000},
    "192.168.2.5": {"Name": "b410-2070s-1","Memory": 32, "GPU": 8000},
    "192.168.2.6": {"Name": "b410-2070s-2","Memory": 32, "GPU": 8000},
    "192.168.2.7": {"Name": "b410-2070s-3","Memory": 32, "GPU": 8000},
    "192.168.2.133": {"Name": "b410-2070s-4","Memory": 64, "GPU": 8000},

}

CLUSTER_RESOURCES = {
    "gpu_mem": 24,  # GB
    "ram": 64
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

    def update_memory_usage(self):
        """更新所有节点的内存剩余情况（单位：GB）"""
        query = '''
            (node_memory_MemAvailable_bytes / 1024^3)
        '''
        results = self._query_prometheus(query)
        print('memory_result', results)
        
        # 遍历集群状态中的每个节点（以IP为键）
        for ip in self.cluster_state:
            # 从查询结果中获取该IP的内存可用值，若不存在则默认为0
            available_gb = results.get(ip, 0)
            
            # 计算已用内存 = 总内存 - 可用内存
            self.cluster_state[ip]["Memory_Used"] = round(
                self.cluster_state[ip]["Memory"] - available_gb, 2
            )
            # 记录可用内存
            self.cluster_state[ip]["Memory_Available"] = round(available_gb, 2)

    def update_gpu_usage(self):
        """更新 GPU 使用率（基于 DCGM_FI_DEV_GPU_UTIL）"""
        # 查询时按 instance 标签分组（IP:Port）
        gpu_values = query_prometheus_gpu()
        for node_gpu in gpu_values:
            query_ip = find_ip_by_name(CLUSTER_INITIAL_STATE, node_gpu["Hostname"])
            self.cluster_state[query_ip]["GPU_Available"] = node_gpu["GPU"]

    def get_status(self):
        """打印当前资源状态（显示节点名称而非IP）"""
        new_memory_resource = 0
        new_gpu_resoure = 0
        for ip, stats in self.cluster_state.items():

            new_gpu_resoure = new_gpu_resoure +  float(stats['Memory_Available'])
            new_memory_resource = new_memory_resource + float(stats['GPU_Available'])

        return new_gpu_resoure, new_memory_resource,self.cluster_state
    
    def run_monitor(self, interval: int = 60):
        """持续监控"""
        while True:
            self.update_memory_usage()
            self.update_gpu_usage()
            new_gpu_resoure, new_memory_resource, stats = self.get_status()
            return new_gpu_resoure, new_memory_resource, stats
    
