import psycopg2

# 连接到 PostgreSQL 数据库
conn = psycopg2.connect(
    dbname="coco",
    user="postgres",
    password="TaskingAI321",
    host="192.168.2.75",
    port="5432"
)

# 创建一个游标对象
cur = conn.cursor()

# SQL 创建 schema 和表语句
create_table_sql = """
-- 创建 schema（如果不存在）
CREATE SCHEMA IF NOT EXISTS coco;

-- 创建 task_results 表
CREATE TABLE IF NOT EXISTS coco.task_results (
    id UUID PRIMARY KEY, -- 唯一主键
    task_id UUID NOT NULL, -- 外键引用 tasks 表
    result_description TEXT, -- 处理结果描述
    result_file_path TEXT, -- 处理结果的文件路径
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 结果生成时间
    FOREIGN KEY (task_id) REFERENCES coco.tasks(task_id) ON DELETE CASCADE
);
"""

# 执行 SQL 语句
cur.execute(create_table_sql)

# 提交事务
conn.commit()

# 关闭游标和连接
cur.close()
conn.close()

print("Tables 'task_results' created successfully in the 'coco' schema!")

