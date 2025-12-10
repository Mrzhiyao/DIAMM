import requests

PROMETHEUS_URL = "http://192.168.2.75:9090/api/v1/query"


# 查询GPU显存使用情况
QUERY = 'gpuram_kB{nvidia_gpu="mem",statistic="gpu"}'

# 查询内存使用情况
QUERY = 'ram_kB{statistic="used"}'

resp = requests.get(PROMETHEUS_URL, params={"query": QUERY})
data = resp.json()

if data["status"] != "success":
    raise RuntimeError(f"Prometheus query failed: {data}")

for result in data["data"]["result"]:
    metric = result["metric"]
    value_kb = float(result["value"][1])
    value_gb = value_kb / (1024 ** 2)   # kB -> GB

    instance = metric.get("instance", "unknown")
    print(f"Instance {instance}: {value_gb:.3f} GB GPU memory used")

# 查询内存使用情况
QUERY = 'ram_kB{statistic="free"}'

resp = requests.get(PROMETHEUS_URL, params={"query": QUERY})
data = resp.json()

if data["status"] != "success":
    raise RuntimeError(f"Prometheus query failed: {data}")

for result in data["data"]["result"]:
    metric = result["metric"]
    value_kb = float(result["value"][1])
    value_gb = value_kb / (1024 ** 2)   # kB -> GB

    instance = metric.get("instance", "unknown")
    print(f"Instance {instance}: {value_gb:.3f} GB memory free")

QUERY = 'ram_kB{statistic="buffers"}'

resp = requests.get(PROMETHEUS_URL, params={"query": QUERY})
data = resp.json()

if data["status"] != "success":
    raise RuntimeError(f"Prometheus query failed: {data}")

for result in data["data"]["result"]:
    metric = result["metric"]
    value_kb = float(result["value"][1])
    value_gb = value_kb / (1024 ** 2)   # kB -> GB

    instance = metric.get("instance", "unknown")
    print(f"Instance {instance}: {value_gb:.3f} GB memory buffers")

QUERY = 'ram_kB{statistic="cached"}'

resp = requests.get(PROMETHEUS_URL, params={"query": QUERY})
data = resp.json()

if data["status"] != "success":
    raise RuntimeError(f"Prometheus query failed: {data}")

for result in data["data"]["result"]:
    metric = result["metric"]
    value_kb = float(result["value"][1])
    value_gb = value_kb / (1024 ** 2)   # kB -> GB

    instance = metric.get("instance", "unknown")
    print(f"Instance {instance}: {value_gb:.3f} GB memory cached")

def query_prometheus_gpu():
    result1 = []
    result2 = []
    result3 = []
    try:
        QUERY = 'ram_kB{statistic="free"}'
        resp = requests.get(PROMETHEUS_URL, params={"query": QUERY})
        data = resp.json()
        if data["status"] == "success":
            for item in data["data"]["result"]:
                metric = item["metric"]
                value_kb = float(item["value"][1])
                # value_gb = value_kb / (1024 ** 2)   # kB -> GB
                value_kb = value_kb / (1024 )   # kB -> GB
                instance = metric.get("instance", "unknown")
                # print(f"Instance {instance}: {value_gb:.3f} GB GPU memory used")
                if instance.replace(':9100', '') == '192.168.2.31':
                    host_name_app = 'b410-jetson-1'
                elif instance.replace(':9100', '') == '192.168.2.32':
                    host_name_app = 'b410-jetson-2'
                elif instance.replace(':9100', '') == '192.168.2.35':
                    host_name_app = 'b410-jetson-5'
                elif instance.replace(':9100', '') == '192.168.2.36':   
                    host_name_app = 'b410-jetson-6'
                result1.append({
                    "Hostname": host_name_app,
                    # "Hostname": instance.replace(':9100', ''),
                    "GPU": value_kb
                })

        QUERY = 'ram_kB{statistic="buffers"}'
        resp = requests.get(PROMETHEUS_URL, params={"query": QUERY})
        data = resp.json()
        if data["status"] == "success":
            for item in data["data"]["result"]:
                metric = item["metric"]
                value_kb = float(item["value"][1])
                # value_gb = value_kb / (1024 ** 2)   # kB -> GB
                value_kb = value_kb / (1024)   # kB -> GB
                instance = metric.get("instance", "unknown")
                # print(f"Instance {instance}: {value_gb:.3f} GB GPU memory used")
                if instance.replace(':9100', '') == '192.168.2.31':
                    host_name_app = 'b410-jetson-1'
                elif instance.replace(':9100', '') == '192.168.2.32':
                    host_name_app = 'b410-jetson-2'
                elif instance.replace(':9100', '') == '192.168.2.35':
                    host_name_app = 'b410-jetson-5'
                elif instance.replace(':9100', '') == '192.168.2.36':   
                    host_name_app = 'b410-jetson-6'
                result2.append({
                    "Hostname": host_name_app,
                    # "Hostname": instance.replace(':9100', ''),
                    "GPU": value_kb
                })

        QUERY = 'ram_kB{statistic="cached"}'
        resp = requests.get(PROMETHEUS_URL, params={"query": QUERY})
        data = resp.json()
        if data["status"] == "success":
            for item in data["data"]["result"]:
                metric = item["metric"]
                value_kb = float(item["value"][1])
                # value_gb = value_kb / (1024 ** 2)   # kB -> GB
                value_kb = value_kb / (1024 )   # kB -> GB
                instance = metric.get("instance", "unknown")
                # print(f"Instance {instance}: {value_gb:.3f} GB GPU memory used")
                if instance.replace(':9100', '') == '192.168.2.31':
                    host_name_app = 'b410-jetson-1'
                elif instance.replace(':9100', '') == '192.168.2.32':
                    host_name_app = 'b410-jetson-2'
                elif instance.replace(':9100', '') == '192.168.2.35':
                    host_name_app = 'b410-jetson-5'
                elif instance.replace(':9100', '') == '192.168.2.36':   
                    host_name_app = 'b410-jetson-6'
                result3.append({
                    "Hostname": host_name_app,
                    # "Hostname": instance.replace(':9100', ''),
                    "GPU": value_kb
                })

        # 如果任何一个结果为空，返回空列表
        if result1 == [] or result2 == [] or result3 == []:
            return []
        
        # 使用字典按 Hostname 累加 GPU 值
        gpu_sum = {}
        for item in result1 + result2 + result3:
            hostname = item["Hostname"]
            if hostname not in gpu_sum:
                gpu_sum[hostname] = 0
            gpu_sum[hostname] += item["GPU"]
        
        # 转换为列表格式
        result = [{"Hostname": hostname, "GPU": gpu_sum[hostname]} for hostname in gpu_sum]
        print('result', result)
        return result

    except Exception as e:
        print(f"请求异常: {e}")
        return None

# query_prometheus_gpu()
