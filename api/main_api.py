import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.database_manager import DatabaseManager
from core.face_processor import FaceProcessor
import uvicorn

# 1. 初始化 FastAPI 实例
app = FastAPI(
    title="SIM-UOW Face Attendance System API",
    description="支持人脸录入、检测与识别的后端接口",
    version="1.0.0"
)

# 2. 配置 CORS (跨域资源共享)
# 必须配置，否则你的 HTML 页面无法通过 JS 访问这个 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 实例化核心 OOP 模块
db = DatabaseManager()
processor = FaceProcessor()


# 辅助函数：将上传的二进制文件转为 OpenCV 格式
async def bytes_to_cv2(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片文件")
    return img


# ------------------------------------------------------------
# 路由 1: 人脸录入 (Face Recording / Registration)
# ------------------------------------------------------------
@app.post("/register")
async def register_face(
        account_id: int = Form(...),
        file: UploadFile = File(...)
):
    """
    接收前端传来的 AccountID 和 照片，提取特征并存入数据库
    """
    img = await bytes_to_cv2(file)

    # 提取特征
    embedding = processor.app.get(img)
    if not embedding:
        return {"success": False, "message": "未检测到人脸，请重新拍摄"}

    # 存入数据库
    if db.connect():
        # 默认取画面中最大的人脸 (embedding[0])
        success = db.save_face_embedding(account_id, embedding[0].normed_embedding)
        db.close()

        if success:
            return {"success": True, "message": f"学生 {account_id} 录入成功"}

    return {"success": False, "message": "数据库写入失败"}


# ------------------------------------------------------------
# 路由 2: 实时识别 (Face Comparison / Attendance)
# ------------------------------------------------------------
@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    """
    接收前端抓拍，返回识别到的学生姓名及相似度
    """
    img = await bytes_to_cv2(file)

    faces = processor.app.get(img)
    if not faces:
        return {"success": True, "identities": []}

    identities = []
    if db.connect():
        for face in faces:
            # 数据库 1:N 检索 (调用我们之前在 db_manager 写的升级版方法)
            name, confidence = db.find_nearest_face(face.normed_embedding)
            if name:
                # 触发签到逻辑
                # db.log_attendance_safe(account_id, session_id=1)
                identities.append({"name": name, "confidence": round(float(confidence), 2)})
            else:
                identities.append({"name": "Unknown", "confidence": 0.0})
        db.close()

    return {"success": True, "identities": identities}


if __name__ == "__main__":
    # 启动服务器，监听 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)