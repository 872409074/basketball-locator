import cv2
from ultralytics import YOLO
from collections import deque

# 1. 加载模型
model = YOLO('D:/runs/detect/train-5/weights/best.pt') 

# 2. 路径设置（必须加 r，且 r 和引号之间不能有空格）
input_video = r'D:\freshman-recruitment-basketball-locator-main\freshman-recruitment-basketball-locator-main\evaluation\basketball_dribble_evaluation.mp4'
output_video = r'D:\freshman-recruitment-basketball-locator-main\result_final.avi'

# 3. 尺寸与帧率
width = 640
height = 480
fps = 30

# 4. 初始化视频写入器（MJPG 兼容性最好）
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ 无法打开视频文件，请检查路径！")
    exit()

print("开始处理视频，请稍候...")

# --- 加分项：轨迹队列 ---
# 保存最近 30 个检测框的中心点，用来画轨迹
track_history = deque(maxlen=30)

# --- 加分项：防抖动（上一帧中心点） ---
last_center = None 

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))

    results = model(frame, conf=0.5) 

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]
            
            # 计算当前帧检测到的中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # --- 加分项：平滑滤波防抖（让框不闪烁，且绝不消失） ---
            if last_center is not None:
                lx, ly = last_center
                # 把当前坐标和上一帧坐标做 7:3 的加权平均
                cx = int(0.7 * cx + 0.3 * lx)
                cy = int(0.7 * cy + 0.3 * ly)
            last_center = (cx, cy)

            # --- 加分项：记录运动轨迹 ---
            track_history.append((cx, cy))

            # 画绿框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # --- 加分项：显示标签、置信度和中心坐标 ---
            label = f"{cls_name} {conf:.2f} | ({cx}, {cy})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- 加分项：绘制历史运动轨迹（蓝色粗线） ---
    if len(track_history) > 1:
        for i in range(1, len(track_history)):
            thickness = int((i / len(track_history)) * 4) + 1
            cv2.line(frame, track_history[i-1], track_history[i], (255, 0, 0), thickness)

    out.write(frame)
    
    # 打印进度
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"已处理 {frame_count} 帧...")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 大功告成！结果已保存在: {output_video}")