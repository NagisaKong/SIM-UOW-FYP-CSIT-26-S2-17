import cv2
import numpy as np
from core.attendance_manager import AttendanceManager


def main():
    # 1. 初始化考勤管理器
    print("🚀 正在初始化考勤系统 (GPU 加速已就绪)...")
    manager = AttendanceManager()

    # 2. 打开摄像头 (1 为 OBS 虚拟摄像头)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ 错误：无法打开 OBS 虚拟摄像头，请确保 OBS 已开启“启动虚拟摄像机”！")
        return

    # 设置窗口为可调节模式
    window_name = 'SIM-UOW Face Attendance System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 初始窗口大小建议 (可以手动拉伸)
    cv2.resizeWindow(window_name, 1280, 720)

    print("✅ 系统已启动！按下 'Q' 键退出程序。")

    frame_count = 0
    last_identities = []  # 存储识别到的结果列表

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        h, w, _ = frame.shape

        # --- 核心逻辑：每 30 帧执行一次深度识别 ---
        if frame_count % 30 == 0:
            # 假设 identify_and_checkin 返回的是识别结果列表
            last_identities = manager.identify_and_checkin(frame)
            print(f"📊 当前识别到: {last_identities}")

        # --- 视觉优化：构建“大画布”显示模式 ---
        # 创建一个比原图更宽的画布 (原图宽度 + 400像素的信息展示区)
        canvas_width = w + 400
        canvas = np.zeros((h, canvas_width, 3), dtype=np.uint8)

        # 1. 将原始摄像头画面贴在画布左侧
        canvas[0:h, 0:w] = frame

        # 2. 在右侧黑色区域绘制分割线
        cv2.line(canvas, (w, 0), (w, h), (50, 50, 50), 2)

        # 3. 在右侧区域绘制识别信息
        margin_left = w + 20
        cv2.putText(canvas, "ATTENDANCE LOG", (margin_left, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        # 循环显示识别到的人名
        y_offset = 90
        if not last_identities:
            cv2.putText(canvas, "Scanning...", (margin_left, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        else:
            # 如果返回的是列表，逐行打印
            if isinstance(last_identities, list):
                for idx, person in enumerate(last_identities):
                    # 根据识别结果显示不同颜色（这里示例用绿色）
                    text = f"{idx + 1}. {person}"
                    cv2.putText(canvas, text, (margin_left, y_offset + (idx * 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # 如果返回的是单条字符串
                cv2.putText(canvas, str(last_identities), (margin_left, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # --- 最终显示 ---
        cv2.imshow(window_name, canvas)

        # 按 Q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 系统已关闭。")


if __name__ == "__main__":
    main()


    #  DATABASE_URL=postgresql://postgres.jbutnqhsgytegjhhftbk:simuowFYPcsit26s217@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres
    # self.app = FaceAnalysis(
    #             name='buffalo_l',
    #             providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    #         )