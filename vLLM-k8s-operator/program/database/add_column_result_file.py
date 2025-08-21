from sqlalchemy import MetaData, Table, text, create_engine

DATABASE_URL = "postgresql+psycopg2://postgres:TaskingAI321@192.168.2.75:5432/coco"

# 创建数据库引擎
engine = create_engine(DATABASE_URL, echo=True)

# 初始化 MetaData 并绑定引擎
metadata = MetaData(schema="coco")
metadata.reflect(bind=engine)  # 使用 reflect 方法加载表信息

# 加载表结构
tasks_table = Table('task_results_file', metadata, autoload_with=engine)

# 检查并添加 vector 列  
if 'vector' not in tasks_table.c:  # 检查列是否已存在  
    with engine.connect() as conn:  
        alter_table_query = text('ALTER TABLE coco.task_results_file ADD COLUMN vector VECTOR(768);')  
        conn.execute(alter_table_query)  # 执行 SQL  
        conn.commit()  # 显式提交事务  
        print("列 vector 已成功添加到 task_results_file 表")  
else:  
    print("列 vector 已存在，无需添加") 

