import os
import ssl
import asyncio
import aiofiles
import yaml
from yaml import SafeLoader
from aiokubernetes import ApiClient, Configuration, CoreV1Api

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

async def async_load_kube_config(config_path: str = "/etc/kubernetes/admin.conf") -> ApiClient:
    """专为Master节点设计的配置加载方法"""
    try:
        # 使用绝对路径避免扩展问题
        expanded_path = os.path.abspath(os.path.expanduser(config_path))
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Kubeconfig文件不存在: {expanded_path}")

        # 读取Master节点的admin.conf
        async with aiofiles.open(expanded_path, mode='r') as f:
            content = await f.read()
            config_data = yaml.load(content, Loader=SafeLoader)

        # 创建SSL上下文（关键修复点）
        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations('/etc/kubernetes/pki/ca.crt')  # 硬编码CA路径

        # 构建配置对象
        config = Configuration()
        cluster = config_data['clusters'][0]['cluster']
        config.host = cluster['server']
        config.ssl_ca_cert = '/etc/kubernetes/pki/ca.crt'  # 强制指定CA路径
        config.cert_file = '/etc/kubernetes/pki/apiserver-kubelet-client.crt'
        config.key_file = '/etc/kubernetes/pki/apiserver-kubelet-client.key'
        config.ssl_context = ssl_context  # 注入自定义SSL上下文

        return ApiClient(configuration=config)
    except Exception as e:
        raise RuntimeError(f"加载Kubernetes配置失败: {str(e)}")

async def find_container_info(
    container_name: str,
    namespace: str = "vllm-k8s-operator-system"
) -> dict:
    """主业务逻辑"""
    api_client = None
    try:
        api_client = await async_load_kube_config()
        v1 = CoreV1Api(api_client)
        
        # 获取Pod列表
        pod_response = await v1.list_namespaced_pod(namespace=namespace)
        found_pod = next(
            (pod for pod in pod_response.obj.items 
             if any(c.name == container_name for c in pod.spec.containers)),
            None
        )
        
        if not found_pod:
            return {"error": f"未找到容器 {container_name}"}
        
        # 获取关联服务
        service_response = await v1.list_service_for_all_namespaces()
        services = [
            svc for svc in service_response.obj.items
            if svc.metadata.namespace == namespace
            and all(
                found_pod.metadata.labels.get(k) == v 
                for k, v in (svc.spec.selector or {}).items()
            )
        ]
        
        return {
            "node": found_pod.spec.node_name,
            "pod": found_pod.metadata.name,
            "services": [
                {
                    "name": svc.metadata.name,
                    "type": svc.spec.type,
                    "ports": [
                        {
                            "service_port": p.port,
                            "target_port": p.target_port,
                            "node_port": p.node_port if svc.spec.type in ["NodePort", "LoadBalancer"] else None
                        } for p in svc.spec.ports
                    ]
                } for svc in services
            ]
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if api_client:
            await api_client.close()

async def main():
    container_result = await find_container_info("stable-diffusion-35")
    print(container_result)
    print(container_result['node'])
    print('ip:', nodes_ip[container_result['node']])
    print('port:', container_result['services'][0]['ports'][0]['node_port'])
    base_url = "http://" + nodes_ip[container_result['node']] + ":" + str(container_result['services'][0]['ports'][0]['node_port']) + "/v1/"
    print("base_url", base_url)

if __name__ == "__main__":
    asyncio.run(main())
