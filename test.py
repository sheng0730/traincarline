import cv2
import numpy as np
from ultralytics import YOLO

# 加载训练好的 YOLOv8 模型
model = YOLO("best.pt")  # 替换为您的 YOLOv8 模型路径

# 定义目标类别（根据您的模型类别索引）
VEHICLE_CLASSES = [0, 1]  # 替换为实际车辆类别索引，例如 0: car, 1: truck

# 输入和输出视频路径
video_input_path = "input_video.mp4"  # 替换为您的输入视频路径
video_output_path = "output_trajectory.mp4"

# 打开输入视频
cap = cv2.VideoCapture(video_input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 配置输出视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# 初始化颜色映射和轨迹字典
np.random.seed(42)
colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(1000)}  # 为每个track_id分配颜色
tracks = {}  # 存储每个track_id的轨迹点

# 主循环：逐帧处理视频
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLOv8 进行目标检测
    results = model(frame)
    for result in results:
        for box in result.boxes:
            # 提取边界框、置信度和类别
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
            conf = box.conf[0]  # 置信度
            class_id = int(box.cls[0])  # 类别ID

            # 仅处理指定类别的对象
            if class_id not in VEHICLE_CLASSES:
                continue

            # 绘制边界框和标签
            color = colors[class_id]
            label = f"{model.names[class_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 更新轨迹
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if class_id not in tracks:
                tracks[class_id] = []
            tracks[class_id].append(center)

    # 绘制轨迹
    for track_id, points in tracks.items():
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], colors[track_id], 2)

    # 保存处理后的视频帧
    out.write(frame)

# 释放资源
cap.release()
out.release()
print(f"Processed video saved as {video_output_path}")