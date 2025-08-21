import requests

PROMETHEUS_URL = "http://192.168.2.75:9090"
QUERY = 'DCGM_FI_DEV_FB_FREE{pod!=""}'

def query_prometheus_gpu():
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": QUERY}
        )
        response.raise_for_status()  # 检查 HTTP 状态码
        
        data = response.json()
        if data["status"] == "success":
            results = data["data"]["result"]
            # print(results)
            outprints = []
            for result in results:
                # print(result) 
                try:
                    outprint  = {
                        "Hostname": result["metric"]["Hostname"],
                        "exported_pod": result["metric"]["exported_pod"],
                        "exported_container": result["metric"]["exported_container"],
                        "GPU": result["value"][1]
                    }
                except:
                    # print(result["metric"]["Hostname"] + '在k8s上无占用显存的程序')
                    outprint  = {
                        "Hostname": result["metric"]["Hostname"],
                        "exported_pod": '',
                        "exported_container": '',
                        "GPU": result["value"][1]
                    }
                # print('output', outprint)
                outprints.append(outprint)
                pod = result["metric"].get("pod", "unknown")
                value = result["value"][1]  # 指标值
                # print(f"Pod: {pod}, GPU利用率: {value}%")
            
            return outprints
        else:
            print("查询失败:", data.get("error", "未知错误"))
            return None
            
    except Exception as e:
        print(f"请求异常: {e}")
        return None

# query_prometheus_gpu()
