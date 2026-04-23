from core.database_manager import DatabaseManager
from core.face_processor import FaceProcessor


class AttendanceManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.processor = FaceProcessor()
        self.is_connected = self.db.connect()

    def identify_and_checkin(self, frame):
        faces = self.processor.app.get(frame)
        if not faces:
            return "Searching..."

        results = []
        for face in faces:
            embedding = face.normed_embedding

            # 这里拿到的 name 就是数据库里的 full_name
            name, confidence = self.db.find_nearest_face(embedding)

            if name:
                # 依然用 AccountID 记考勤（后台逻辑不变）
                # 这里如果你想更严谨，可以让 find_nearest_face 把 ID 也一起返回
                # 为了演示简单，我们先专注于显示名字
                results.append(f"{name} ({confidence:.2f})")
            else:
                results.append("Unknown")

        return results