import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceProcessor:
    def __init__(self, ctx_id=0, det_size=(640, 640)):
        # 初始化模型 (只加载一次)
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_embedding(self, image_path):
        """输入图片路径，返回 512维特征向量"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            return None

        # 默认只取画面中最大的一张脸
        # sorted_faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].normed_embedding