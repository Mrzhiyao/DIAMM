import requests
import time
PROMETHEUS_URL = "http://192.168.2.75:9090"
QUERY = 'kube_pod_info'

def query_pods_info():
    try:
        # 修改后的 Prometheus 查询（包含 container 标签）⇩⇩⇩
        # QUERY_WITH_CONTAINER = 'sum(container_memory_usage_bytes{container!="", container!="POD"}) by (node, pod, container)'
        QUERY_WITH_CONTAINER = '''
        sum(container_memory_usage_bytes{
            container!="", 
            container!="POD",
            namespace="vllm-k8s-operator-system"  # 添加命名空间过滤
        }) by (node, pod, container, namespace)    # 包含 namespace 分组
        '''
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
    
