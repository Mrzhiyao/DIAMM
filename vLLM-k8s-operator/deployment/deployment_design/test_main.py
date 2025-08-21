from prome_resource import ClusterMonitor
import requests
import time
from query_pods import query_pods_info_k8s
from datetime import datetime, timedelta
import re

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

PROMETHEUS_URL = "http://192.168.2.75:9090"
QUERY = 'kube_pod_info'

def query_pods_t2v():
    try:
        # 修改后的 Prometheus 查询（包含 container 标签）⇩⇩⇩
        QUERY_WITH_CONTAINER = 'sum(container_memory_usage_bytes{container!="", container!="POD"}) by (node, pod, container)'
        
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={
                "query": QUERY_WITH_CONTAINER,  # 使用包含容器名称的查询
                "time": time.time()  # 可选时间戳参数
            }
        )
        response.raise_for_status()

        data = response.json()
        if data["status"] == "success":
            results = data["data"]["result"]
            container_info = []
            
            for result in results:
                metric = result["metric"]
                try:
                    # 提取节点、Pod、容器三要素 ⇩⇩⇩
                    record = {
                        "Hostname": metric["node"],
                        "pod": metric["pod"],
                        "container": metric["container"],  # 新增容器名称字段
                        "value": float(result["value"][1])  # 示例数值字段
                    }
                except KeyError as e:
                    print(f"缺少关键标签 {e}, 原始数据: {metric}")
                    continue
                
                container_info.append(record)
            
            return container_info
        else:
            print("查询失败:", data.get("error", "未知错误"))
            return None

    except Exception as e:
        print(f"请求异常: {e}")
        return None


# def query_pods_t2v():
#     try:
#         # 动态时间窗口（例如：只保留过去5分钟内的活跃数据）
#         time_threshold = datetime.utcnow() - timedelta(minutes=2)
#         timestamp_threshold = time_threshold.timestamp()

#         # 修改查询语句：添加时间戳筛选
#         QUERY = f'''
#         sum(
#             container_memory_usage_bytes{{container!="", container!="POD"}}
#             and on(pod) timestamp(container_memory_usage_bytes) > {timestamp_threshold}
#         ) by (node, pod, container)
#         '''

#         response = requests.get(
#             f"{PROMETHEUS_URL}/api/v1/query",
#             params={"query": QUERY}
#         )
#         response.raise_for_status()

#         data = response.json()
#         if data["status"] == "success":
#             results = data["data"]["result"]
#             container_info = []
            
#             for result in results:
#                 metric = result["metric"]
#                 try:
#                     # 提取必要字段（已通过 PromQL 过滤过期数据）
#                     record = {
#                         "Hostname": metric["node"],
#                         "pod": metric["pod"],
#                         "container": metric["container"],
#                         "value": float(result["value"][1]),
#                         "last_seen": datetime.fromtimestamp(float(result["value"][0]))
#                     }
#                 except KeyError as e:
#                     print(f"缺少关键标签 {e}, 原始数据: {metric}")
#                     continue
                
#                 container_info.append(record)
            
#             return container_info
#         else:
#             print("查询失败:", data.get("error", "未知错误"))
#             return None

#     except Exception as e:
#         print(f"请求异常: {e}")
#         return None


def find_gpu_nodes(data, min_gpu):
    """
    基于GPU可用量筛选节点的函数
    :param data: 集群节点信息字典
    :param min_gpu: 最低要求的可用GPU显存(MB)
    :return: 符合要求的节点IP列表及详细资源信息
    """
    qualified_nodes = []
    
    for ip, info in data.items():
        # 转换GPU可用值为整数
        gpu_available = int(info['GPU_Available'])

        # 资源校验与筛选
        if gpu_available >= min_gpu:
            # print(ip, gpu_available)
            node_info = {
                'IP': ip,
                'Hostname': info['Name'],
                'Total_GPU': info['GPU'],
                'Available_GPU': gpu_available,
                'GPU_Utilization': f"{(1 - gpu_available/info['GPU'])*100:.1f}%",
                'Memory_Status': f"{info['Memory_Used']}GB/{info['Memory']}GB"
            }
            qualified_nodes.append(node_info)
    
    # 按GPU可用量降序排序
    return sorted(qualified_nodes, key=lambda x: x['Available_GPU'], reverse=True)


def extract_hostname_and_container_number(data):
    result = []
    for item in data:
        hostname = item['Hostname']
        pod = item['pod']
        container_str = item['container']
        # 匹配末尾连续数字
        match = re.search(r'(\d+)$', container_str)
        last_number = int(match.group(1)) if match else None
        result.append({
            'Hostname': hostname,
            'container_last_number': last_number,
            'pod':pod
        })
    return result

def available_ip(threshold):
    monitor = ClusterMonitor()
    new_gpu_resoure, new_memory_resource,stats = monitor.run_monitor()
    result = find_gpu_nodes(stats, threshold)
    service_develpoment = query_pods_t2v()
    filtered = [item for item in service_develpoment if item['pod'] != '']
    filtered_text2video_items = [
        item for item in filtered
    if "text2video"  in item.get("pod", "").lower()
    ]
    # print('sss, ', filtered_text2video_items)
    querys = extract_hostname_and_container_number(filtered_text2video_items)
    text2video_pods_ip = []
    for query in querys:
        pods_status = check_pod_status(query['pod'], 'vllm-k8s-operator-system')
        print(pods_status[0])
        if pods_status[0] == True:
            text2video_pods_ip.append(query['Hostname'])
         
    # exclude_hostnames = {item['Hostname'] for item in filtered_text2video_items}
    exclude_hostnames = {item for item in text2video_pods_ip}

    service_content = query_pods_info_k8s()
    # print('k8s_service_content', service_content)
    loss_gpu = []
    for item in service_content:
        if 'text2video' in item:
            # print('text2video', item)
            loss_gpu.append(item['Hostname'])    

    # 过滤list1中不包含这些hostname的条目
    filtered_result = [item for item in result if item['Hostname'] not in exclude_hostnames]
    for node in filtered_result:
        if node['Hostname'] in loss_gpu:
            if node['Available_GPU']>5000:
                node['Available_GPU'] = node['Available_GPU'] - 19000  # 保持字符串类型
                if node['Available_GPU'] <0:
                    node['Available_GPU'] = 0

        # print(f"IP: {node['IP']} | 主机: {node['Hostname']}")
        # print(f"  可用显存: {node['Available_GPU']}MB (总{node['Total_GPU']}MB)")
        # print(f"  显存利用率: {node['GPU_Utilization']} | 内存使用: {node['Memory_Status']}\n")

    
    excluded_hostnames = {"b410-4090d-2", "b410-2070s-4"}
    filtered_result = [
        item for item in filtered_result 
        if item["Hostname"] not in excluded_hostnames
    ]
    # 屏蔽指定节点192.168.2.190：
    filtered_result = [item for item in filtered_result if item["Hostname"] not in excluded_hostnames]

    # for node in filtered_result:
    #     if node['IP'] == '192.168.2.190':
    #         node.update({
    #             'Available_GPU': 10,
    #             'GPU_Utilization': '99.9%'
    #         })
    #         break  # 找到后立即退出循环
    

    # print('ips:', filtered_result)
    return filtered_result



from kubernetes import client, config

def check_pod_status(pod_name, namespace="default"):
    """检查 Pod 是否处于 Running 状态且容器就绪"""
    config.load_kube_config()  # 如果运行在集群内，使用 config.load_incluster_config()
    v1 = client.CoreV1Api()
    
    try:
        pod = v1.read_namespaced_pod(pod_name, namespace)
        # 检查 Pod 状态是否为 Running
        if pod.status.phase != "Running":
            return False, f"Pod 处于 {pod.status.phase} 状态"
        # 检查所有容器是否就绪
        for container_status in pod.status.container_statuses:
            if not container_status.ready:
                return False, f"容器 {container_status.name} 未就绪"
        return True, "Pod 运行正常"
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return False, "Pod 不存在"
        else:
            return False, f"API 错误: {e.reason}"

# available_ips = available_ip(5000)
# sorted_available_ips = sorted(available_ips, key=lambda x: x['Total_GPU'], reverse=True)
# hostnames = [node['Hostname'] for node in sorted_available_ips]
# print(hostnames)

# monitor = ClusterMonitor()
# new_gpu_resoure, new_memory_resource,stats = monitor.run_monitor()
# # print(new_gpu_resoure, new_memory_resource)

# # print(stats)

# # 使用示例：查找可用GPU≥5000MB的节点
# threshold = 5000
# result = find_gpu_nodes(stats, threshold)
# service_develpoment = query_pods_t2v()
# filtered = [item for item in service_develpoment if item['pod'] != '']
# filtered_text2video_items = [
#     item for item in filtered
# if "text2video"  in item.get("pod", "").lower()
# ]
# # 提前预留的gpu需要删去
# print(filtered_text2video_items)
# print(result)
# exclude_hostnames = {item['Hostname'] for item in filtered_text2video_items}

# # 过滤list1中不包含这些hostname的条目
# filtered_result = [item for item in result if item['Hostname'] not in exclude_hostnames]

# # 格式化输出
# print(f"可用GPU≥{threshold}MB的节点:")
# for node in filtered_result:
#     print(f"IP: {node['IP']} | 主机: {node['Hostname']}")
#     print(f"  可用显存: {node['Available_GPU']}MB (总{node['Total_GPU']}MB)")
#     print(f"  显存利用率: {node['GPU_Utilization']} | 内存使用: {node['Memory_Status']}\n")
