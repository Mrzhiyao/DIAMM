import yaml
import random
import argparse
import re 

def sanitize_container_name(model_name: str, suffix: str) -> str:
    """生成符合 Kubernetes 规范的容器名称"""
    # Step 1: 转换全小写
    cleaned = model_name.lower()
    
    # Step 2: 替换非法字符为连字符
    cleaned = re.sub(r'[^a-z0-9-]', '-', cleaned)
    
    # Step 3: 合并连续连字符
    cleaned = re.sub(r'-+', '-', cleaned)
    
    # Step 4: 去除首尾连字符
    cleaned = cleaned.strip('-')
    
    # Step 5: 限制长度并拼接后缀
    max_base_length = 30  # 保留后缀空间
    final_name = f"{cleaned[:max_base_length]}-{suffix}"
    
    # 最终再次清理
    final_name = re.sub(r'[^a-z0-9-]', '', final_name)[:63]
    return final_name

def modify_text2text_yaml_template(
    input_file,
    output_file,
    model_name="Qwen2.5-7B-Instruct",
    node_name="b410-3090-1",
    x=None
):
    """修改YAML模板的核心函数"""
    
    # 读取YAML文件
    with open(input_file, 'r', encoding='utf-8') as f:
        docs = list(yaml.safe_load_all(f))  # 支持多文档YAML

    # 生成随机数x（如果未提供）
    x = x if x is not None else random.randint(0, 100)
    
    # 替换 Deployment 相关字段
    deployment = docs[0]
    deployment["metadata"]["name"] = f"text2text-{x}"
    deployment["spec"]["selector"]["matchLabels"]["app"] = f"text2text-{x}"
    deployment["spec"]["template"]["metadata"]["labels"]["app"] = f"text2text-{x}"
    for container in deployment['spec']['template']['spec']['containers']:
        if container['name'] == 'text2text-x':
            container['name'] = sanitize_container_name(model_name, x)
            break
    
    # 替换容器中的模型路径
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    container["volumeMounts"][0]["mountPath"] = f"/root/.cache/modelscope/{model_name}"
    container["volumeMounts"][0]["subPath"] = model_name
    container["command"][2] = container["command"][2].replace(
        "Qwen2.5-7B-Instruct", model_name
    )
    
    # 替换节点名称
    deployment["spec"]["template"]["spec"]["nodeName"] = node_name
    
    # 替换 Service 相关字段
    service = docs[1]
    service["metadata"]["name"] = f"text2text-{x}-service"
    service["spec"]["selector"]["app"] = f"text2text-{x}"
    
    # 生成新的NodePort（保持前三位不变，替换最后两位）
    original_port = service["spec"]["ports"][0]["nodePort"]
    new_port = (original_port // 100) * 100 + x  # 例如 31113 -> 31100 + x
    service["spec"]["ports"][0]["nodePort"] = new_port

    # 保存修改后的YAML
    with open(output_file, 'w') as f:
        yaml.dump_all(docs, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated config: {output_file} (x={x}, node_port={new_port})")

    return output_file

def local_modify_text2text_yaml_template(
    input_file,
    output_file,
    model_name="Qwen2.5-7B-Instruct",
    node_name="b410-3090-1",
    x=None
):
    """修改YAML模板的核心函数"""
    
    # 读取YAML文件
    with open(input_file, 'r', encoding='utf-8') as f:
        docs = list(yaml.safe_load_all(f))  # 支持多文档YAML

    # 生成随机数x（如果未提供）
    x = x if x is not None else random.randint(0, 100)
    
    # 替换 Deployment 相关字段
    deployment = docs[0]
    deployment["metadata"]["name"] = f"text2text-{x}"
    deployment["spec"]["selector"]["matchLabels"]["app"] = f"text2text-{x}"
    deployment["spec"]["template"]["metadata"]["labels"]["app"] = f"text2text-{x}"
    for container in deployment['spec']['template']['spec']['containers']:
        if container['name'] == 'text2text-x':
            container['name'] = sanitize_container_name(model_name, x)
            break
    
    # 替换容器中的模型路径
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    container["volumeMounts"][0]["mountPath"] = f"/root/.cache/modelscope/{model_name}"
    # container["volumeMounts"][0]["subPath"] = model_name

    # 替换容器中的模型路径
    container_local = deployment["spec"]["template"]["spec"]
    print(container_local)
    container_local["volumes"][0]["hostPath"]["path"] = container_local["volumes"][0]["hostPath"]["path"] + model_name

    container["command"][2] = container["command"][2].replace(
        "Qwen2.5-7B-Instruct", model_name
    )
    
    # 替换节点名称
    deployment["spec"]["template"]["spec"]["nodeName"] = node_name
    
    # 替换 Service 相关字段
    service = docs[1]
    service["metadata"]["name"] = f"text2text-{x}-service"
    service["spec"]["selector"]["app"] = f"text2text-{x}"
    
    # 生成新的NodePort（保持前三位不变，替换最后两位）
    original_port = service["spec"]["ports"][0]["nodePort"]
    new_port = (original_port // 100) * 100 + x  # 例如 31113 -> 31100 + x
    service["spec"]["ports"][0]["nodePort"] = new_port

    # 保存修改后的YAML
    with open(output_file, 'w') as f:
        yaml.dump_all(docs, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated config: {output_file} (x={x}, node_port={new_port})")

    return output_file

if __name__ == "__main__":
    # 命令行参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="local-text2text-x.yaml", help="输入模板文件路径")
    parser.add_argument("--output", default="local-text2text-x-development.yaml", help="输出文件路径")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct", help="自定义模型名称")
    parser.add_argument("--node", default="b410-4090d-2", help="自定义节点名称")
    parser.add_argument("--x", type=int, help="手动指定x值（可选）")
    args = parser.parse_args()
    local_modify_text2text_yaml_template(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        node_name=args.node,
        x=args.x
    )
