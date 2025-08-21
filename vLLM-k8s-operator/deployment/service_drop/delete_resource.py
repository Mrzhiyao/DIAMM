from kubernetes import client, config
from kubernetes.client.rest import ApiException

def delete_by_container_names(container_names, namespace="vllm-k8s-operator-system"):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()

    # 通过容器名称查找关联的Pod
    all_pods = v1.list_namespaced_pod(namespace=namespace).items
    target_pods = []
    
    # 第一步：根据容器名筛选Pod
    for pod in all_pods:
        # 检查所有容器名称
        container_names_in_pod = [c.name for c in pod.spec.containers]
        # 检查init容器（如果需要的话）
        # container_names_in_pod += [c.name for c in pod.spec.init_containers or []]
        
        # 判断是否有容器名匹配
        if any(name in container_names for name in container_names_in_pod):
            target_pods.append(pod.metadata.name)

    # 第二步：执行删除逻辑（复用原有逻辑）
    # print('delete container', target_pods)
    delete_pods_and_deployments(target_pods, namespace)



def delete_pods_and_deployments(pod_names, namespace):
    # 加载Kubernetes配置（默认读取~/.kube/config）

    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()

    for pod_name in pod_names:
        try:
            # 获取Pod详细信息
            pod = v1.read_namespaced_pod(name=pod_name, namespace="vllm-k8s-operator-system")
            # print('pod', pod)
            # 查找关联的Deployment
            deployment_name = None
            
            # 通过OwnerReferences追溯上级资源
            for owner in pod.metadata.owner_references or []:
                if owner.kind == "ReplicaSet":
                    # 获取ReplicaSet详细信息
                    rs = apps_v1.read_namespaced_replica_set(
                        name=owner.name, 
                        namespace="vllm-k8s-operator-system"
                    )
                    # 从ReplicaSet获取Deployment名称
                    for rs_owner in rs.metadata.owner_references or []:
                        if rs_owner.kind == "Deployment":
                            deployment_name = rs_owner.name
                            break
                    if deployment_name:
                        break
            if deployment_name:
                # 删除Deployment
                print(f"Deleting deployment {deployment_name}")
                apps_v1.delete_namespaced_deployment(
                    name=deployment_name,
                    namespace="vllm-k8s-operator-system",
                    body=client.V1DeleteOptions(
                        propagation_policy="Background",
                        grace_period_seconds=5
                    )
                )
                print(f"✅ Deployment {deployment_name} deleted")
            else:
                # 直接删除Pod
                print(f"Deleting orphan pod {pod_name}")
                v1.delete_namespaced_pod(
                    name=pod_name,
                    namespace="vllm-k8s-operator-system",
                    body=client.V1DeleteOptions(
                        grace_period_seconds=5
                    )
                )
                print(f"✅ Pod {pod_name} deleted")

        except ApiException as e:
            print(f"❌ Error processing {pod_name}: {e.reason}")


if __name__ == "__main__":
    containers_to_delete = ["stable-diffusion-9"] 
    delete_by_container_names(containers_to_delete)

