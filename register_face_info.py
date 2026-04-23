from core.database_manager import DatabaseManager
from core.face_processor import FaceProcessor

def run_registration(student_account_id, photo_path):
    # 1. 初始化
    db = DatabaseManager()
    processor = FaceProcessor()

    print(f"⏳ 正在处理学号对应账号 ID: {student_account_id} 的人脸...")

    # 2. 提取特征
    try:
        embedding = processor.extract_embedding(photo_path)
        if embedding is None:
            print("❌ 未在图片中检测到人脸，录入失败。")
            return
    except Exception as e:
        print(f"❌ 特征提取出错: {e}")
        return

    # 3. 存入数据库
    if db.connect():
        success = db.save_face_embedding(student_account_id, embedding)
        if success:
            print(f"✅ 录入成功！人脸向量已安全存入 Supabase 云端。")
        db.close()

if __name__ == "__main__":
    # 测试一下：假设数据库里已经有了 AccountID 为 1 的学生
    # 你可以换成你项目目录下的图片文件名
    run_registration(student_account_id=2, photo_path="yuqian.jpg")