import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import time
from sqlalchemy import text

def search_result(db_conn, select_id):
    """
    同步版本：查询向量数据库并返回匹配的视频片段
    
    Args:
        task: 文本查询任务
        db_conn: PostgreSQL 数据库连接对象
        process_text_sync: 同步文本处理函数，返回特征向量
        top_n: 返回的结果数量
        
    Returns:
        list: 匹配的视频片段列表
    """
    # print('select_id', select_id)
    try:
        # 构建查询
        query = sql.SQL("""
            SELECT 
                task_id,
                result_description,
                result_file_path,
                created_at
            FROM coco.task_results
            WHERE task_id = %s;
        """)
        
        # 执行查询
        with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (select_id, ))
            results = cur.fetchall()
        # print(results)

        return results
    
    except Exception as e:
        print(f"数据库操作失败: {str(e)}")
        db_conn.rollback()  # 出错回滚事务
        raise
    finally:
        # 注意：这里不关闭连接，由调用者管理连接生命周期
        pass
