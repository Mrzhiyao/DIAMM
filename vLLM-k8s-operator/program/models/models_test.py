import os
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import BYTEA  # 用于二进制数据
from pgvector.sqlalchemy import Vector  # pgvector 扩展的类型

from .config import SQLALCHEMY_PGDATABASE_URL

# 数据库目录不存在的时候自动创建目录
folder_path = os.path.dirname(SQLALCHEMY_PGDATABASE_URL.replace("postgresql://", ""))
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 本地扫描数据库
BaseModel = declarative_base()
engine = create_engine(SQLALCHEMY_PGDATABASE_URL)
DatabaseSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# PexelsVideo数据库
BaseModelPexelsVideo = declarative_base()
engine_pexels_video = create_engine('postgresql://postgres:TaskingAI321@192.168.2.75/PexelsVideo')  # 修改为 PostgreSQL 连接
DatabaseSessionPexelsVideo = sessionmaker(autocommit=False, autoflush=False, bind=engine_pexels_video)

def create_tables():
    """创建数据库表"""
    BaseModel.metadata.create_all(bind=engine)
    BaseModelPexelsVideo.metadata.create_all(bind=engine_pexels_video)

# 通过basemodel绑定的表创建
class Image(BaseModel):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True)
    path = Column(String(4096), index=True)  # 文件路径
    modify_time = Column(DateTime)  # 文件修改时间
    features = Column(Vector(512))  # 使用 pgvector 存储向量，768 为向量维度

class Video(BaseModel):
    __tablename__ = "video"
    id = Column(Integer, primary_key=True)
    path = Column(String(4096), index=True)  # 文件路径
    frame_time = Column(Integer, index=True)  # 这一帧所在的时间
    modify_time = Column(DateTime)  # 文件修改时间
    features = Column(Vector(512))  # 使用 pgvector 存储向量，768 为向量维度

class PexelsVideo(BaseModelPexelsVideo):
    __tablename__ = "PexelsVideo"
    id = Column(Integer, primary_key=True)
    title = Column(String(128))  # 标题
    description = Column(String(256))  # 视频描述
    duration = Column(Integer, index=True)  # 视频时长，单位秒
    view_count = Column(Integer, index=True)  # 视频播放量
    thumbnail_loc = Column(String(256), index=True)  # 视频缩略图链接
    content_loc = Column(String(256))  # 视频链接
    thumbnail_feature = Column(Vector(512))  # 使用 pgvector 存储缩略图特征向量

