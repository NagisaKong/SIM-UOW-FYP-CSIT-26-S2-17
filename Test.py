import cv2
import numpy as np
import warnings
import time  # 引入 time 库，用来在画面上显示比对时间

# 💡 祖传补丁区
np.int = int  # 修复 numpy 的老版本 int 报错
warnings.filterwarnings("ignore", category=FutureWarning)  # 屏蔽第三方库烦人的 FutureWarning 警告

from insightface.app import FaceAnalysis


# ==========================================
# 核心算法：计算余弦相似度 (Cosine Similarity)
# ==========================================
def compute_similarity(feat1, feat2):
    """计算两个 512 维向量的相似度，越接近 1 越相似"""
    # 确保输入是 numpy 数组且维度正确
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))


print("⏳ 正在初始化 AI 模型...")
# 1. 加载模型 (开启 GPU 加速)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 代表调用你的显卡加速

# ==========================================
# 💡 关键步骤：准备本地底库
# ==========================================
# 请确保项目根目录下有你要比对的本地照片，并在此处修改文件名
# 比如：zhangsan.jpg, lisi.jpg 等。为了演示，我们假设拍到的是自己，
# 找一张自己不同时期或不同角度清晰正面照作为底库。
base_img_path = '人山人海.jpg'
print(f"📂 正在加载本地底库照片: [{base_img_path}]...")

# 使用 np.imdecode 绕过中文路径问题读取底库图
img_base = cv2.imdecode(np.fromfile(base_img_path, dtype=np.uint8), -1)

if img_base is None:
    print(f"❌ 无法加载底库图片，请检查文件 {base_img_path} 是否在正确的位置。")
    # 如果找不到底库照片，我们可以用一个默认的陌生人照片来演示陌生人拒绝
    print("💡 为了演示，我将加载一张郭德纲的照片作为示例陌生人底库。请确保项目目录下有 guodegang.jpg")
    base_img_path = 'guodegang.jpg'
    img_base = cv2.imdecode(np.fromfile(base_img_path, dtype=np.uint8), -1)
    if img_base is None:
        print(f"❌ 默认陌生人底库 guodegang.jpg 也未找到！请至少准备一张照片。")
        exit()

print(f"🧠 正在提取本地底库人脸特征...")
faces_base = app.get(img_base)

if len(faces_base) == 0:
    print(f"❌ 在底库图 {base_img_path} 中未检测到人脸！请换一张清晰的人脸照片作为底库。")
    exit()

# 缓存底库第一张人脸的 512 维向量
feat_base_cached = faces_base[0].normed_embedding
print("✅ 底库特征提取完成并缓存！")

# ==========================================
# 2. 准备实时摄像头比对
# ==========================================
print("🎥 正在请求打开摄像头...")
# 打开摄像头 (参数 0 通常指笔记本自带的默认摄像头)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ 无法打开摄像头，请检查设备连接或 Windows 隐私权限设置！")
    exit()

print("✅ 摄像头已开启，实时比对中...")
print("👉 操作说明：将焦点放在弹出的视频窗口上。按键盘 'q' 键退出。")

# 设定识别阈值
THRESHOLD = 0.45

while True:
    # 读取最新的一帧画面
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法获取画面流")
        break

    # 3. 让 AI 实时检测当前帧中的人脸
    faces_curr = app.get(frame)

    # 4. 对当前帧检测到的人脸进行实时比对和绘制
    if len(faces_curr) > 0:
        for face in faces_curr:
            # 提取当前人脸特征
            feat_curr = face.normed_embedding

            # 计算与底库特征的相似度
            sim_score = compute_similarity(feat_base_cached, feat_curr)

            # 获取人脸框坐标
            box = face.bbox.astype(int)

            # 根据得分和阈值判断结果
            text = f"Sim: {sim_score:.2f}"
            color = (0, 0, 255)  # 默认为红色 (验证失败)

            if sim_score >= THRESHOLD:
                text = f"Passed! Sim: {sim_score:.2f}"
                color = (0, 255, 0)  # 绿色 (验证通过)
                # 可以在通过时顺便显示一个认证标志
                cv2.putText(frame, "VERIFIED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 在当前帧上画框和写结果
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    else:
        # 如果未检测到人脸，可以显示一个提示
        cv2.putText(frame, "No Face Detected", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # 5. 显示实时比对画面
    cv2.imshow('FYP Smart Vision - Live Match', frame)

    # 监听键盘按键 (1 毫秒刷新一次)，按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 释放资源，优雅退出
cap.release()
cv2.destroyAllWindows()
print("👋 测试结束，摄像头已关闭。")