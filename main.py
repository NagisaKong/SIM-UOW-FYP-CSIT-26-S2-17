import cv2
import time
from core.attendance_manager import AttendanceManager


def main():
    # 1. 初始化
    print("🚀 正在初始化考勤系统...")
    manager = AttendanceManager()

    # 2. 打开摄像头 (确认 1 是 OBS 虚拟摄像头)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ 错误：无法打开 OBS 虚拟摄像头，请确保 OBS 已开启“启动虚拟摄像机”！")
        return

    print("✅ 无感考勤系统已启动，正在捕捉 OBS 画面...")

    frame_count = 0
    last_identity = "Scanning..."

    cv2.namedWindow('SIM-UOW System', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 暂时无法获取画面...")
            continue

        frame_count += 1

        # 每 30 帧比对一次，防止系统卡死
        if frame_count % 30 == 0:
            last_identity = manager.identify_and_checkin(frame)
            # 在控制台打印，方便我们调试阈值
            print(f"📊 识别结果: {last_identity}")

        # 在窗口左上角实时显示识别状态
        cv2.putText(frame, f"Identity: {last_identity}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow('SIM-UOW Attendance System (Press Q to Quit)', frame)

        # 按 Q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()