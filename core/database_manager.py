import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

class DatabaseManager:
    def __init__(self):
        load_dotenv()
        self.db_url = os.getenv("DATABASE_URL")
        self.conn = None

    def connect(self):
        """建立连接并注册向量插件"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            register_vector(self.conn)
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False

    def save_face_embedding(self, account_id, embedding, model_name="arcface"):
        """将 512维向量 存入 FACE_EMBEDDING 表"""
        sql = """
        INSERT INTO FACE_EMBEDDING (AccountID, embedding_vector, model_name, model_version, dimension)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                # 这里的 embedding 应该是 numpy 数组
                cur.execute(sql, (account_id, embedding, model_name, 'r100', 512))
                self.conn.commit()
                return True
        except Exception as e:
            print(f"❌ 存入向量失败: {e}")
            self.conn.rollback()
            return False

    def find_nearest_face(self, target_embedding, threshold=0.6):
        """
        升级版：不仅找 ID，还把名字一起带回来
        """
        distance_threshold = 1 - threshold

        # 使用 JOIN 关联 PERSONAL_INFO 表
        sql = """
              SELECT f.AccountID, p.full_name, (f.embedding_vector <=> %s) AS distance
              FROM FACE_EMBEDDING f
                       JOIN PERSONAL_INFO p ON f.AccountID = p.AccountID
              WHERE f.is_active = TRUE
              ORDER BY distance ASC
              LIMIT 1; \
              """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (target_embedding,))
                result = cur.fetchone()

                # 如果找到了且距离在范围内
                if result and result[2] < distance_threshold:
                    return result[1], 1 - result[2]  # ✅ 返回名字和相似度
                return None, None
        except Exception as e:
            print(f"❌ 数据库关联查询出错: {e}")
            return None, None
    def log_attendance(self, account_id, session_id=1):
        """记录考勤到 ATTENDANCE_RECORD 表"""
        sql = """
              INSERT INTO ATTENDANCE_RECORD (AttendanceSessionID, AccountID, status)
              VALUES (%s, %s, 'present') ON CONFLICT (AttendanceSessionID, AccountID) DO NOTHING; \
              """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (session_id, account_id))
                self.conn.commit()
                return True
        except Exception as e:
            print(f"❌ 记录考勤失败: {e}")
            self.conn.rollback()
            return False

    def close(self):
        if self.conn:
            self.conn.close()