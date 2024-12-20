import cv2
from ultralytics import YOLO
import supervision as sv
from supervision.detection.core import Detections
from supervision.draw.color import ColorPalette
from scipy.spatial import distance

# 加载 YOLO 模型
model = YOLO("yolov8n.pt")  # 替换为您训练好的权重文件

# 定义车辆类别（COCO 数据集中车辆类别的索引）
VEHICLE_CLASSES = [2, 3, 5, 7]  # 'car', 'motorcycle', 'bus', 'truck'

# 输入视频路径
video_input_path = "input_video.mp4"  # 替换为您的输入视频路径
# 输出视频路径
video_output_path = "output_trajectory.mp4"

# 打开视频文件
cap = cv2.VideoCapture(video_input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 配置输出视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# 初始化车辆轨迹存储
vehicle_tracks = {}
track_colors = ColorPalette()

# 欧几里得距离计算，用于判断最近轨迹
def get_nearest_track(center, tracks, threshold=50):
    nearest_idx = None
    min_distance = threshold
    for idx, points in tracks.items():
        last_point = points[-1]
        dist = distance.euclidean(center, last_point)
        if dist < min_distance:
            min_distance = dist
            nearest_idx = idx
    return nearest_idx

# 主循环：逐帧处理视频
track_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 进行目标检测
    results = model(frame)
    detections = Detections.from_yolov8(results[0])

    # 筛选车辆类别的检测结果
    vehicle_detections = []
    for i, class_id in enumerate(detections.class_id):
        if class_id in VEHICLE_CLASSES:
            vehicle_detections.append(detections[i])

    # 获取车辆中心点
    centers = []
    for det in vehicle_detections:
        x1, y1, x2, y2 = det.xyxy
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        centers.append(center)

    # 更新轨迹
    for center in centers:
        nearest_idx = get_nearest_track(center, vehicle_tracks)
        if nearest_idx is not None:
            vehicle_tracks[nearest_idx].append(center)
        else:
            vehicle_tracks[track_id] = [center]
            track_id += 1

    # 绘制轨迹
    for idx, points in vehicle_tracks.items():
        color = track_colors(idx)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 2)

    # 绘制检测框
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{model.names[class_id]} {conf:.2f}"
        for _, _, conf, class_id, _ in vehicle_detections
    ]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # 保存处理后的视频帧
    out.write(frame)

# 释放资源
cap.release()
out.release()
print(f"Processed video saved as {video_output_path}")
