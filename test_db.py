import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

# 1. 加载 .env 文件中的安全链接
load_dotenv()
db_url = os.getenv("DATABASE_URL")


def test_supabase_connection():
    print("⏳ 正在尝试连接 Supabase 云端数据库...")
    try:
        # 2. 建立连接
        conn = psycopg2.connect(db_url)

        # 3. 注册向量插件 (为存入 512维 数组做准备)
        register_vector(conn)

        # 4. 执行一个简单的查询来验证
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            db_version = cur.fetchone()

            print("\n✅ ==== 连接成功！ ====")
            print(f"📡 数据库信息: {db_version[0]}")

        # 5. 关闭连接
        conn.close()
        print("🔒 连接已安全关闭。")

    except Exception as e:
        print(f"\n❌ 连接失败，请检查网络或密码！错误详情：\n{e}")


if __name__ == "__main__":
    test_supabase_connection()