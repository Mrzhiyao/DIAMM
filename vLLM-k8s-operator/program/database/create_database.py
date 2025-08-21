import psycopg2

# 连接到默认的 PostgreSQL 数据库（通常是 "postgres"）
conn = psycopg2.connect(
    dbname="postgres",  # 先连接默认数据库
    user="postgres",
    password="TaskingAI321",
    host="192.168.2.75",
    port="5432"
)

conn.autocommit = True  # 必须开启自动提交，否则创建数据库会失败
cur = conn.cursor()

# 检查数据库是否已存在
cur.execute("SELECT 1 FROM pg_database WHERE datname = 'coco'")
exists = cur.fetchone()

if not exists:
    cur.execute("CREATE DATABASE coco")
    print("Database 'coco' created successfully.")
else:
    print("Database 'coco' already exists.")

# 关闭游标和连接
cur.close()
conn.close()

