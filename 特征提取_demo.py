import cv2
from insightface.app import FaceAnalysis

# 1. 初始化 InsightFace 模型
# 注意：ctx_id=0 表示使用第一块 GPU。如果你的电脑没配好 CUDA，或者想用 CPU 跑，把 ctx_id 改成 -1
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. 指定你想测试的单张图片路径
# 请将这里的路径替换为你电脑里实际存在的图片路径
image_path = "capture.jpg"

print(f"正在读取图片: {image_path} ...")
img = cv2.imread(image_path)

if img is None:
    print("❌ 错误: 无法读取图像，请检查图片路径是否正确！")
else:
    # 3. 检测人脸并提取特征
    print("正在进行人脸检测与特征提取...")
    faces = app.get(img)

    # 4. 判断结果并打印
    if len(faces) == 0:
        print("⚠️ 警告: 图片中未检测到任何人脸！")
    elif len(faces) > 1:
        print(f"⚠️ 警告: 图片中检测到了 {len(faces)} 张人脸，提取脚本建议使用单人清晰正面照！")
    else:
        # 成功检测到单张人脸，提取归一化特征向量
        feat_vector = faces[0].normed_embedding

        print("\n✅ ==== 特征提取成功！ ====")
        print(f"数据类型: {type(feat_vector)}")
        print(f"向量维度: {feat_vector.shape} (证明这是标准的 512 维特征)")
        print("\n👇 ==== 具体的 512 维特征数据如下 ==== 👇")

        # 打印完整的 numpy 数组
        print(feat_vector)

        # 如果你想看它转成普通 Python 列表（List）的样子，可以取消下面这行的注释
        # print(feat_vector.tolist())