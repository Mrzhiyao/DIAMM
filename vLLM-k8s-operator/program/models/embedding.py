# # 模型性能基准测试
# import time
# import numpy as np
# import torch
# from PIL import Image
# from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
# import logging
# from .config import *
# import traceback
# from .search import search_image_by_feature

# logger = logging.getLogger(__name__)

# device = "cuda"  # 推理设备，可选cpu、cuda、mps
# # DEVICE = "cuda"
# input_image = "./program/models/test.png"  # 测试图片。图片大小影响速度，一般相机照片为4000x3000。图片内容不影响速度。
# input_text = "dog"  # 测试文本


# print("Loading models...")
# clip_model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_NAME)
# clip_processor = AutoProcessor.from_pretrained(MODEL_NAME)
# print("Models loaded.")

# # 图像处理性能基准测试
# print("*" * 50)

# logger.info("Loading model...")
# # 图像
# model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_NAME).to(torch.device(DEVICE))
# processor = AutoProcessor.from_pretrained(MODEL_NAME)
# logger.info("Model loaded.")

# def process_text(input_text):
#     """
#     预处理文字，返回文字特征
#     :param input_text: string, 被处理的字符串
#     :return: <class 'numpy.nparray'>,  文字特征
#     """
#     feature = None
#     if not input_text:
#         return None
#     try:
#         text = processor(text=input_text[0:77], return_tensors="pt", padding=True)["input_ids"].to(torch.device(DEVICE))
#         feature = model.get_text_features(text).detach().cpu().numpy()
#     except Exception as e:
#         logger.warning(f"处理文字报错：{repr(e)}")
#         traceback.print_stack()
#     return feature


# def get_image_data(path: str, ignore_small_images: bool = True):
#     """
#     获取图片像素数据，如果出错返回 None
#     :param path: string, 图片路径
#     :param ignore_small_images: bool, 是否忽略尺寸过小的图片
#     :return: <class 'numpy.nparray'>, 图片数据，如果出错返回 None
#     """
#     import os  
#     # image = Image.open(path)
#     try:
#         image = Image.open(path)
#         if ignore_small_images:
#             width, height = image.size
#             if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
#                 return None
#                 # processor 中也会这样预处理 Image
#         # 在这里提前转为 np.array 避免到时候抛出异常
#         image = image.convert('RGB')
#         image = np.array(image)
#         return image
#     except Exception as e:
#         logger.warning(f"打开图片报错：{path} {repr(e)}")
#         return None


# def process_image(path, ignore_small_images=True):
#     """
#     处理图片，返回图片特征
#     :param path: string, 图片路径
#     :param ignore_small_images: bool, 是否忽略尺寸过小的图片
#     :return: <class 'numpy.nparray'>, 图片特征
#     """
#     image = get_image_data(path, ignore_small_images)
#     if image is None:
#         return None
#     feature = get_image_feature(image)
#     return feature


# def get_image_feature(images):
#     """
#     :param images: 图片
#     :return: feature
#     """
#     feature = None
#     try:
#         inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(torch.device(DEVICE))
#         feature = model.get_image_features(inputs).detach().cpu().numpy()
#     except Exception as e:
#         logger.warning(f"处理图片报错：{repr(e)}")
#         traceback.print_stack()
#     return feature



import aiohttp

import json

API_BASE = "http://192.168.2.75:7879"  # 替换为实际服务地址

async def process_text(text: str) -> list:
    """调用文本处理API（异步版本）"""
    url = f"{API_BASE}/process_text"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"text": text},
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["feature"]
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return None

async def process_image(image_path: str) -> list:
    """调用图像处理API（异步版本）"""
    url = f"{API_BASE}/process_image"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"image_path": image_path},
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["feature"]
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return None
