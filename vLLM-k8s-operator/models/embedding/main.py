# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import torch
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
import os
from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', "openai/clip-vit-large-patch14-336")  # CLIP模型
DEVICE = os.getenv('DEVICE', 'cuda')  # 推理设备，cpu/cuda/mps，建议先跑benchmark.py看看cpu还是显卡速度更快。因为数据搬运也需要时间，所以不一定是GPU更快。

app = FastAPI()

# 全局模型变量（单例）
_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_path: str

@app.on_event("startup")
def startup_event():
    """服务启动时加载模型（仅执行一次）"""
    global _MODEL, _PROCESSOR
    try:
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_NAME)
        _MODEL = _MODEL.to(_DEVICE).eval()
        logging.info("模型加载成功")
    except Exception as e:
        logging.error(f"模型初始化失败: {str(e)}")
        raise

@app.post("/process_text")
async def process_text_api(request: TextRequest):
    """文本处理API"""
    try:
        inputs = _PROCESSOR(text=request.text[:77], return_tensors="pt", padding=True).to(_DEVICE)
        with torch.no_grad():
            features = _MODEL.get_text_features(**inputs)
        return {"feature": features.cpu().numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_image")
async def process_image_api(request: ImageRequest):
    """图像处理API"""
    try:
        image = Image.open(request.image_path)
        inputs = _PROCESSOR(images=image, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            features = _MODEL.get_image_features(**inputs)
        return {"feature": features.cpu().numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7879)
