import datetime
import logging
import os

from sqlalchemy import asc
from sqlalchemy.orm import Session

# from models import Image, Video, PexelsVideo
from .models_test import Image, Video, PexelsVideo
# from sqlalchemy.dialects.postgresql import ARRAY, FLOAT
# from sqlalchemy import cast
# from sqlalchemy import func, and_, not_
import psycopg2
import numpy as np
import ast
logger = logging.getLogger(__name__)


def get_image_features_by_id(session: Session, image_id: int):
    """
    返回id对应的图片feature
    """
    # features = session.query(Image.features).filter_by(id=image_id).first()

    conn = psycopg2.connect(
        dbname="video",
        user="postgres",
        password="TaskingAI321",
        host="192.168.2.75",
        port="5432"
    )
    cur = conn.cursor()

    query = """
    SELECT features
    FROM image
    WHERE id = %s
    """
    params = [image_id]  # Start with the positive feature

    cur.execute(query, params)
    return_features = cur.fetchall()
    cur.close()
    conn.close()
    features_list = ast.literal_eval(return_features[0][0])
    # print('return_features', type(features_list), [features_list])
    if not return_features:
        logger.warning("用数据库的图来进行搜索，但id在数据库中不存在")
        return None
    return [features_list]


def get_image_path_by_id(session: Session, id: int):
    """
    返回id对应的图片路径
    """
    path = session.query(Image.path).filter_by(id=id).first()
    if not path:
        return None
    return path[0]


def get_image_count(session: Session):
    """获取图片总数"""
    return session.query(Image).count()


def delete_image_if_outdated(session: Session, path: str) -> bool:
    """
    判断图片是否修改，若修改则删除
    :param session: Session, 数据库 session
    :param path: str, 图片路径
    :return: bool, 若文件未修改返回 True
    """
    record = session.query(Image).filter_by(path=path).first()
    if not record:
        return False
    modify_time = os.path.getmtime(path)
    modify_time = datetime.datetime.fromtimestamp(modify_time)
    if record.modify_time == modify_time:
        logger.debug(f"文件无变更，跳过：{path}")
        return True
    logger.info(f"文件有更新：{path}")
    session.delete(record)
    session.commit()
    return False


def delete_video_if_outdated(session: Session, path: str) -> bool:
    """
    判断视频是否修改，若修改则删除
    :param session: Session, 数据库 session
    :param path: str, 视频路径
    :return: bool, 若文件未修改返回 True
    """
    record = session.query(Video).filter_by(path=path).first()
    if not record:
        return False
    modify_time = os.path.getmtime(path)
    modify_time = datetime.datetime.fromtimestamp(modify_time)
    if record.modify_time == modify_time:
        logger.debug(f"文件无变更，跳过：{path}")
        return True
    logger.info(f"文件有更新：{path}")
    session.query(Video).filter_by(path=path).delete()
    session.commit()
    return False


def get_video_paths(session: Session, filter_path: str = None):
    """获取所有视频的路径，支持通过路径筛选"""
    query = session.query(Video.path).distinct()
    print('query video', query)
    if filter_path:
        query = query.filter(Video.path.like("%" + filter_path + "%"))
    for i, in query:
        yield i


def get_frame_times_features_by_path(session: Session, path: str):
    """获取路径对应视频的features"""
    l = (
        session.query(Video.frame_time, Video.features)
        .filter_by(path=path)
        .order_by(Video.frame_time)
        .all()
    )
    frame_times, features = zip(*l)
    return frame_times, features


def get_features_times_path(session: Session, path: str):
    """获取路径对应视频的features"""
    l = (
        session.query(Video.frame_time, Video.features)
        .filter_by(path=path)
        .order_by(Video.frame_time)
        .all()
    )
    frame_times, features = zip(*l)
    return frame_times, features


def get_video_count(session: Session):
    """获取视频总数"""
    return session.query(Video.path).distinct().count()


def get_pexels_video_count(session: Session):
    """获取视频总数"""
    return session.query(PexelsVideo).count()


def get_video_frame_count(session: Session):
    """获取视频帧总数"""
    return session.query(Video).count()


def delete_video_by_path(session: Session, path: str):
    """删除路径对应的视频数据"""
    session.query(Video).filter_by(path=path).delete()
    session.commit()


def add_image(session: Session, path: str, modify_time: datetime.datetime, features: list[float]):
    """添加图片到数据库"""
    logger.info(f"新增文件：{path}")
    image = Image(path=path, modify_time=modify_time, features=features)
    session.add(image)
    session.commit()


def add_video(session: Session, path: str, modify_time, frame_time_features_generator):
    """
    将处理后的视频数据入库
    :param session: Session, 数据库session
    :param path: str, 视频路径
    :param modify_time: datetime, 文件修改时间
    :param frame_time_features_generator: 返回(帧序列号,特征)元组的迭代器
    """
    # 使用 bulk_save_objects 一次性提交，因此处理至一半中断不会导致下次扫描时跳过
    logger.info(f"新增文件：{path}")
    video_list = (
        Video(
            path=path, modify_time=modify_time, frame_time=frame_time, features=features
        )
        for frame_time, features in frame_time_features_generator
    )
    session.bulk_save_objects(video_list)
    session.commit()


def add_pexels_video(session: Session, content_loc: str, duration: int, view_count: int, thumbnail_loc: str, title: str, description: str,
                     thumbnail_feature: bytes):
    """添加pexels视频到数据库"""
    pexels_video = PexelsVideo(
        content_loc=content_loc, duration=duration, view_count=view_count, thumbnail_loc=thumbnail_loc, title=title, description=description,
        thumbnail_feature=thumbnail_feature
    )
    session.add(pexels_video)
    session.commit()


def delete_record_if_not_exist(session: Session, assets: set):
    """
    删除不存在于 assets 集合中的图片 / 视频的数据库记录
    """
    for file in session.query(Image):
        if file.path not in assets:
            logger.info(f"文件已删除：{file.path}")
            session.delete(file)
    for path in session.query(Video.path).distinct():
        path = path[0]
        if path not in assets:
            logger.info(f"文件已删除：{path}")
            session.query(Video).filter_by(path=path).delete()
    session.commit()


def is_video_exist(session: Session, path: str):
    """判断视频是否存在"""
    video = session.query(Video).filter_by(path=path).first()
    if video:
        return True
    return False


def is_pexels_video_exist(session: Session, thumbnail_loc: str):
    """判断pexels视频是否存在"""
    video = session.query(PexelsVideo).filter_by(thumbnail_loc=thumbnail_loc).first()
    if video:
        return True
    return False


def get_image_id_path_features(session: Session) -> tuple[list[int], list[str], list[bytes]]:
    """
    获取全部图片的 id, 路径, 特征，返回三个列表
    """
    session.query(Image).filter(Image.features.is_(None)).delete()
    session.commit()
    query = session.query(Image.id, Image.path, Image.features)
    try:
        id_list, path_list, features_list = zip(*query)
        return id_list, path_list, features_list
    except ValueError:  # 解包失败
        return [], [], []


def get_image_id_path_features_filter_by_path_time(session: Session, path: str, start_time: int, end_time: int, positive_feature: str, negative_feature, positive_threshold, negative_threshold, top_n):
    print('top', top_n)
    print(positive_threshold, negative_threshold)
    """
    根据路径和时间，筛选出对应图片的 id, 路径, 特征，返回三个列表
    """
    positive_threshold = 1-positive_threshold/100
    negative_threshold = 1-negative_threshold/100
    print('positive_threshold', positive_threshold)
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="coco",
        user="postgres",
        password="TaskingAI321",
        host="8",
        port="5432"
    )
    cur = conn.cursor()
    # print('time:', start_time, end_time)
    if positive_feature is not None:
        # positive_feature = str(positive_feature[0])
        positive_feature = f"[{', '.join(map(str, np.array(positive_feature[0], dtype=np.float32).flatten()))}]"
        # print('positive_feature', positive_feature)
        query = """
        SELECT id, path, features, features <=> %s AS positive_distance
        FROM image_infor
        WHERE 1=1  -- This allows for easy addition of conditions
        """

        params = [positive_feature]  # Start with the positive feature

        # Only add the path condition if it's provided
        if path:
            query += " AND path LIKE %s"
            params.append(f"%{path}%")

        # Only add the modify_time condition if start_time and end_time are provided
        if start_time is not None and start_time != 0:
            query += " AND modify_time >= TO_TIMESTAMP(%s)"
            params.append(start_time)

        if end_time is not None and end_time != 0:
            query += " AND modify_time <= TO_TIMESTAMP(%s)"
            params.append(end_time)

        # Only add the positive feature threshold condition if it's provided
        # print(positive_threshold == None)
        if positive_threshold is not None:
            query += " AND features <=> %s <= %s"  # Adjust this if positive_feature can be None
            params.append(positive_feature)
            params.append(positive_threshold)

        # Only add the negative feature threshold condition if it's provided
        if negative_threshold is not None:
            query += " AND features <=> %s >= %s"  # Adjust this if negative_feature can be None
            params.append(positive_feature)
            params.append(negative_threshold)

        # Add the order and limit clause
        query += " ORDER BY positive_distance LIMIT %s;"
        params.append(top_n)

        # Execute the query with the constructed parameters
        # print(query)
        cur.execute(query, params)
        positive_results = cur.fetchall()

        ids, paths, features, positive_scores = [], [], [], []

        # print(positive_results[0])
        for result in positive_results:
            ids.append(result[0])
            paths.append(result[1])
            features.append(result[2])
            positive_scores.append(1-result[3])
        
    

    if negative_feature is not None:
        # positive_feature_str = f"[{', '.join(map(str, np.array(positive_feature, dtype=np.float32).flatten()))}]"
        negative_feature = f"[{', '.join(map(str, np.array(negative_feature[0], dtype=np.float32).flatten()))}]"
        # negative_feature = str(negative_feature[0])
        query_negative = """
        SELECT id, path, features, features <=> %s AS negative_distance
        FROM image
        WHERE 1=1  -- This allows for easy addition of conditions
        """
        
        params_negative = [negative_feature]  # Start with the positive feature

        # Only add the path condition if it's provided
        if path:
            query_negative += " AND path LIKE %s"
            params_negative.append(f"%{path}%")

        # Only add the modify_time condition if start_time and end_time are provided
        if start_time is not None and start_time != 0:
            query_negative += " AND modify_time >= TO_TIMESTAMP(%s)"
            params_negative.append(start_time)

        if end_time is not None and end_time != 0:
            query_negative += " AND modify_time <= TO_TIMESTAMP(%s)"
            params_negative.append(end_time)

        # Only add the positive feature threshold condition if it's provided
        # print(positive_threshold == None)
        if positive_threshold is not None:
            query_negative += " AND features <=> %s <= %s"  # Adjust this if positive_feature can be None
            params_negative.append(negative_feature)
            params_negative.append(positive_threshold)

        # Only add the negative feature threshold condition if it's provided
        if negative_threshold is not None:
            query_negative += " AND features <=> %s >= %s"  # Adjust this if negative_feature can be None
            params_negative.append(negative_feature)
            params_negative.append(negative_threshold)

        # Add the order and limit clause
        query_negative += " ORDER BY negative_distance LIMIT %s;"
        params_negative.append(top_n)

        # Execute the query with the constructed parameters
        cur.execute(query_negative, params_negative)
        negative_results = cur.fetchall()

        negative_ids, negative_paths, negative_features, negative_scores = [], [], [], []
        for result in negative_results:
            negative_ids.append(result[0])
            negative_paths.append(result[1])
            negative_features.append(result[2])
            negative_scores.append(result[3])
        print('negative_results',negative_scores)

        # 合并和去除重叠部分
        # 构建最终结果，排除负向匹配的id
        output_ids, output_paths, output_features, output_scores = [], [], [], []
        n = 0
        for id in ids:
            if id not in negative_ids:
                output_ids.append(ids[n])
                output_paths.append(paths[n])
                output_features.append(features[n])
                output_scores.append(1-positive_scores[n])  # 保存正向相似度分数
            n = n + 1
        print('scores', output_scores)
        cur.close()
        conn.close()
    
        return output_ids, output_paths, output_features, output_scores

    cur.close()
    conn.close()
    print('positive_scores', positive_scores)
    return ids, paths, features, positive_scores



def search_image_by_path(session: Session, path: str):
    """
    根据路径搜索图片
    :return: (图片id, 图片路径) 元组列表
    """
    return (
        session.query(Image.id, Image.path)
        .filter(Image.path.like("%" + path + "%"))
        .order_by(asc(Image.path))
        .all()
    )


def search_video_by_path(session: Session, path: str):
    """
    根据路径搜索视频
    """
    return (
        session.query(Video.path)
        .distinct()
        .filter(Video.path.like("%" + path + "%"))
        .order_by(asc(Video.path))
        .all()
    )


def get_pexels_video_features(session: Session):
    """返回所有pexels视频"""
    query = session.query(
        PexelsVideo.thumbnail_feature, PexelsVideo.thumbnail_loc, PexelsVideo.content_loc,
        PexelsVideo.title, PexelsVideo.description, PexelsVideo.duration, PexelsVideo.view_count
    ).all()
    try:
        thumbnail_feature_list, thumbnail_loc_list, content_loc_list, title_list, description_list, duration_list, view_count_list = zip(*query)
        return thumbnail_feature_list, thumbnail_loc_list, content_loc_list, title_list, description_list, duration_list, view_count_list
    except ValueError:  # 解包失败
        return [], [], [], [], [], [], []


def get_pexels_video_by_id(session: Session, uuid: str):
    """根据id搜索单个pexels视频"""
    return session.query(PexelsVideo).filter_by(id=uuid).first()




def get_images_from_coco(session: Session, path: str, start_time: int, end_time: int, positive_feature: str, negative_feature, positive_threshold, negative_threshold, top_n):
    print('top', top_n)
    print(positive_threshold, negative_threshold)
    """
    根据路径和时间，筛选出对应图片的 id, 路径, 特征，返回三个列表
    """
    positive_threshold = 1-positive_threshold/100
    negative_threshold = 1-negative_threshold/100
    print('positive_threshold', positive_threshold)
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="video",
        user="postgres",
        password="TaskingAI321",
        host="192.168.2.75",
        port="5432"
    )
    cur = conn.cursor()
    # print('time:', start_time, end_time)
    if positive_feature is not None:
        # positive_feature = str(positive_feature[0])
        positive_feature = f"[{', '.join(map(str, np.array(positive_feature[0], dtype=np.float32).flatten()))}]"
        # print('positive_feature', positive_feature)
        query = """
        SELECT id, path, features, features <=> %s AS positive_distance
        FROM image
        WHERE 1=1  -- This allows for easy addition of conditions
        """

        params = [positive_feature]  # Start with the positive feature

        # Only add the path condition if it's provided
        if path:
            query += " AND path LIKE %s"
            params.append(f"%{path}%")

        # Only add the modify_time condition if start_time and end_time are provided
        if start_time is not None and start_time != 0:
            query += " AND modify_time >= TO_TIMESTAMP(%s)"
            params.append(start_time)

        if end_time is not None and end_time != 0:
            query += " AND modify_time <= TO_TIMESTAMP(%s)"
            params.append(end_time)

        # Only add the positive feature threshold condition if it's provided
        # print(positive_threshold == None)
        if positive_threshold is not None:
            query += " AND features <=> %s <= %s"  # Adjust this if positive_feature can be None
            params.append(positive_feature)
            params.append(positive_threshold)

        # Only add the negative feature threshold condition if it's provided
        if negative_threshold is not None:
            query += " AND features <=> %s >= %s"  # Adjust this if negative_feature can be None
            params.append(positive_feature)
            params.append(negative_threshold)

        # Add the order and limit clause
        query += " ORDER BY positive_distance LIMIT %s;"
        params.append(top_n)

        # Execute the query with the constructed parameters
        # print(query)
        cur.execute(query, params)
        positive_results = cur.fetchall()

        ids, paths, features, positive_scores = [], [], [], []

        # print(positive_results[0])
        for result in positive_results:
            ids.append(result[0])
            paths.append(result[1])
            features.append(result[2])
            positive_scores.append(1-result[3])
        
    

    if negative_feature is not None:
        # positive_feature_str = f"[{', '.join(map(str, np.array(positive_feature, dtype=np.float32).flatten()))}]"
        negative_feature = f"[{', '.join(map(str, np.array(negative_feature[0], dtype=np.float32).flatten()))}]"
        # negative_feature = str(negative_feature[0])
        query_negative = """
        SELECT id, path, features, features <=> %s AS negative_distance
        FROM image
        WHERE 1=1  -- This allows for easy addition of conditions
        """
        
        params_negative = [negative_feature]  # Start with the positive feature

        # Only add the path condition if it's provided
        if path:
            query_negative += " AND path LIKE %s"
            params_negative.append(f"%{path}%")

        # Only add the modify_time condition if start_time and end_time are provided
        if start_time is not None and start_time != 0:
            query_negative += " AND modify_time >= TO_TIMESTAMP(%s)"
            params_negative.append(start_time)

        if end_time is not None and end_time != 0:
            query_negative += " AND modify_time <= TO_TIMESTAMP(%s)"
            params_negative.append(end_time)

        # Only add the positive feature threshold condition if it's provided
        # print(positive_threshold == None)
        if positive_threshold is not None:
            query_negative += " AND features <=> %s <= %s"  # Adjust this if positive_feature can be None
            params_negative.append(negative_feature)
            params_negative.append(positive_threshold)

        # Only add the negative feature threshold condition if it's provided
        if negative_threshold is not None:
            query_negative += " AND features <=> %s >= %s"  # Adjust this if negative_feature can be None
            params_negative.append(negative_feature)
            params_negative.append(negative_threshold)

        # Add the order and limit clause
        query_negative += " ORDER BY negative_distance LIMIT %s;"
        params_negative.append(top_n)

        # Execute the query with the constructed parameters
        cur.execute(query_negative, params_negative)
        negative_results = cur.fetchall()

        negative_ids, negative_paths, negative_features, negative_scores = [], [], [], []
        for result in negative_results:
            negative_ids.append(result[0])
            negative_paths.append(result[1])
            negative_features.append(result[2])
            negative_scores.append(result[3])
        print('negative_results',negative_scores)

        # 合并和去除重叠部分
        # 构建最终结果，排除负向匹配的id
        output_ids, output_paths, output_features, output_scores = [], [], [], []
        n = 0
        for id in ids:
            if id not in negative_ids:
                output_ids.append(ids[n])
                output_paths.append(paths[n])
                output_features.append(features[n])
                output_scores.append(1-positive_scores[n])  # 保存正向相似度分数
            n = n + 1
        print('scores', output_scores)
        cur.close()
        conn.close()
    
        return output_ids, output_paths, output_features, output_scores

    cur.close()
    conn.close()
    print('positive_scores', positive_scores)
    return ids, paths, features, positive_scores
