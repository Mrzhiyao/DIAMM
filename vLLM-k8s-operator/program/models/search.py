import base64
import logging
import time
from functools import lru_cache

import numpy as np

from .config import *
from .dataset_operate import (
    get_image_id_path_features_filter_by_path_time,
    get_image_features_by_id,
    get_video_paths,
    get_frame_times_features_by_path,
    get_pexels_video_features,
    get_images_from_coco
)
# from models_test import DatabaseSession, DatabaseSessionPexelsVideo
# from process_assets import match_batch, process_image, process_text


logger = logging.getLogger(__name__)

def search_image_by_feature(
        SessionLocal, 
        positive_feature=None,
        negative_feature=None,
        positive_threshold=POSITIVE_THRESHOLD,
        negative_threshold=NEGATIVE_THRESHOLD,
        path="",
        start_time=None, 
        end_time=None,
        top_n=6
        
):
    """
    通过特征搜索图片，使用PGVector
    """
    t0 = time.time()
    ids, paths, features, scores = get_image_id_path_features_filter_by_path_time(SessionLocal, path, start_time, end_time, positive_feature, negative_feature, positive_threshold, negative_threshold, top_n)
    print('lens', scores, len(ids), len(scores))
    return_list = []
    for id, path, score in zip(ids, paths, scores):
        if not score:
            continue
        return_list.append({
            "url": "api/get_image/%d" % id,
            "path": path,
            "score": float(score),
        })
    # return_list = sorted(return_list, key=lambda x: x["score"], reverse=True)
    logger.info("查询使用时间：%.2f" % (time.time() - t0))
    return ids, paths, features, scores




