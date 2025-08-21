from fastapi import FastAPI  
from fastapi.staticfiles import StaticFiles  
from fastapi.middleware.cors import CORSMiddleware  
import os  

app = FastAPI()  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["*"],  # 允许所有域名的请求  
    allow_credentials=True,  
    allow_methods=["*"],  # 允许所有 HTTP 方法  
    allow_headers=["*"],  # 允许所有请求头  
)  

# 假设图像存储在这些目录  
TRAIN2017_FOLDER = '/yaozhi/vLLM-k8s-operator/train2017/'  
USER_TASK_FOLDER = '/yaozhi/vLLM-k8s-operator/user_tasks/'  
OUTPUT_TASK_FOLDER = '/yaozhi/vLLM-k8s-operator/outputs/' 

# 确保目录存在  
os.makedirs(TRAIN2017_FOLDER, exist_ok=True)  
os.makedirs(USER_TASK_FOLDER, exist_ok=True)  

# 设置静态文件服务  
app.mount("/train2017", StaticFiles(directory=TRAIN2017_FOLDER), name="train2017")  
app.mount("/user_tasks", StaticFiles(directory=USER_TASK_FOLDER), name="user_tasks")  
app.mount("/outputs", StaticFiles(directory=OUTPUT_TASK_FOLDER), name="outputs")  

if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run(app, host="192.168.2.75", port=12365)
