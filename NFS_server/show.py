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
TRAIN2017_FOLDER = '/nfs/ai/datasets/train2017/'  
USER_TASK_FOLDER = '/nfs/ai/datasets/user_tasks/'  
OUTPUT_TASK_FOLDER = '/nfs/ai/datasets/outputs/' 
OUTPUT_VIDEO_FOLDER = '/nfs/ai/ai-model/CogVideo-1.0/outputs/'
DATA_VIDEO_FOLDER = '/nfs/ai/datasets/processed_videos/video_segments/'
DATA_VIDEO_IMG_FOLDER = '/nfs/ai/datasets/processed_videos/key_frames/'

# 确保目录存在  
os.makedirs(TRAIN2017_FOLDER, exist_ok=True)  
os.makedirs(USER_TASK_FOLDER, exist_ok=True)  
os.makedirs(OUTPUT_TASK_FOLDER, exist_ok=True)  
os.makedirs(OUTPUT_VIDEO_FOLDER, exist_ok=True)  
os.makedirs(DATA_VIDEO_FOLDER, exist_ok=True)  
os.makedirs(DATA_VIDEO_IMG_FOLDER, exist_ok=True)  

# 设置静态文件服务  
app.mount("/train2017", StaticFiles(directory=TRAIN2017_FOLDER), name="train2017")  
app.mount("/user_tasks", StaticFiles(directory=USER_TASK_FOLDER), name="user_tasks")  
app.mount("/outputs", StaticFiles(directory=OUTPUT_TASK_FOLDER), name="outputs")  
app.mount("/video_outputs", StaticFiles(directory=OUTPUT_VIDEO_FOLDER), name="video_outputs")  
app.mount("/data_video", StaticFiles(directory=DATA_VIDEO_FOLDER), name="data_video")  
app.mount("/data_video_img", StaticFiles(directory=DATA_VIDEO_IMG_FOLDER), name="data_video_img")  

if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run(app, host="192.168.2.78", port=12365)
