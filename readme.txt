训练：
yolo task=detect \
mode=train \
model=yolov8n.pt \
data=/home/rong/Pointnet_Pointnet2_pytorch/YOLO/tennis_dataset/tennis.yaml \
epochs=100 \
imgsz=640 \
batch=16 \
name=tennis_track_train

##
    task=detect: 任务类型为目标检测。

    mode=train: 模式为训练。

    model=yolov8n.pt: 使用预训练的YOLOv8纳米模型。

    data=...: 指向你刚刚创建的 tennis.yaml 文件。

    epochs=100: 训练100轮。你可以根据损失曲线调整这个数值。

    imgsz=640: 输入图像的尺寸为640x640像素。

    batch=16: 每批处理16张图片。如果GPU内存不足（报CUDA out of memory错误），请减小这个数值，例如改为 batch=8 或 batch=4。

    name=tennis_track_train: 为这次训练运行命名，输出会保存在 runs/detect/tennis_track_train/ 目录下。
##

验证：
yolo task=detect \
mode=val \
model=/home/rong/Pointnet_Pointnet2_pytorch/YOLO/runs/detect/tennis_track_train/weights/best.pt \
data=/home/rong/Pointnet_Pointnet2_pytorch/YOLO/tennis_dataset/tennis.yaml


追踪：
yolo task=detect \
mode=track \
model=/home/rong/Pointnet_Pointnet2_pytorch/YOLO/runs/detect/tennis_track_train/weights/best.pt \
source="/home/rong/Pointnet_Pointnet2_pytorch/YOLO/tenniball and label/追踪原视频.mp4" \
show=True \
tracker=botsort.yaml

##
    mode=track: 模式为追踪（=检测+追踪）。

    model=...: 指向你训练得到的最佳模型 best.pt。

    source=...: 指向你的输入视频文件路径（例如 /home/rong/videos/tennis_match.mp4）。

    show=True: 在屏幕上实时显示追踪结果（如果是在有图形界面的机器上）。对于服务器，可以设置为 False。

    tracker=botsort.yaml: 指定追踪算法。Ultralytics YOLO默认集成BoT-SORT和ByteTrack，推荐先尝试 botsort.yaml。
##




