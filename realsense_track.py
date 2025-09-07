import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import time

class RealSenseTennisTracker:
    def __init__(self, model_path, use_depth=False):
        """
        初始化RealSense网球追踪器
        
        Args:
            model_path: 训练好的YOLO模型路径
            use_depth: 是否使用深度流
        """
        # 加载训练好的YOLO模型
        self.model = YOLO(model_path)
        
        # 配置RealSense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 启用彩色流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.use_depth = use_depth
        if use_depth:
            # 启用深度流
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # 对齐深度帧到彩色帧
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None
        
        # 启动管道
        try:
            self.profile = self.pipeline.start(self.config)
            
            if use_depth:
                # 获取深度传感器的深度尺度
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                print(f"深度尺度: {self.depth_scale}")
            else:
                self.depth_scale = None
                
            print("RealSense摄像头初始化完成")
            
        except RuntimeError as e:
            print(f"无法启动RealSense管道: {e}")
            print("请检查摄像头连接或尝试重新插拔")
            raise e
    
    def process_frame(self, color_frame, depth_frame=None):
        """
        处理每一帧图像，进行网球检测和追踪
        """
        # 将彩色帧转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        
        # 使用YOLO进行追踪
        results = self.model.track(
            source=color_image,
            conf=0.5,  # 置信度阈值
            tracker='botsort.yaml',  # 使用BoT-SORT追踪器
            verbose=False,  # 不输出详细信息
            persist=True  # 保持追踪状态
        )
        
        # 获取深度数据（如果启用深度流）
        depth_data = None
        if depth_frame and self.use_depth:
            depth_data = np.asanyarray(depth_frame.get_data())
        
        # 在图像上绘制结果
        annotated_frame = results[0].plot()
        
        # 如果有检测结果，显示额外信息
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                
                # 计算网球中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # 在中心点画圆
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # 显示追踪ID和置信度
                label = f"ID:{track_id} Conf:{conf:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 如果启用深度，计算距离
                if depth_data is not None and center_y < depth_data.shape[0] and center_x < depth_data.shape[1]:
                    depth_value = depth_data[center_y, center_x] * self.depth_scale
                    distance_text = f"Distance: {depth_value:.2f}m"
                    cv2.putText(annotated_frame, distance_text, (int(x1), int(y1)-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return annotated_frame
    
    def run(self):
        """
        主循环，实时处理视频流
        """
        try:
            print("开始实时追踪，按 'q' 键退出...")
            
            while True:
                try:
                    # 等待一组帧（颜色和深度），增加超时时间
                    frames = self.pipeline.wait_for_frames(10000)  # 10秒超时
                    
                    if self.use_depth:
                        # 对齐深度帧到彩色帧
                        aligned_frames = self.align.process(frames)
                        # 获取对齐后的彩色帧和深度帧
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()
                    else:
                        # 仅获取彩色帧
                        color_frame = frames.get_color_frame()
                        depth_frame = None
                    
                    if not color_frame:
                        print("未获取到彩色帧，继续等待...")
                        time.sleep(0.1)
                        continue
                    
                    # 处理当前帧
                    processed_frame = self.process_frame(color_frame, depth_frame)
                    
                    # 显示结果
                    cv2.imshow('RealSense Tennis Tracking', processed_frame)
                    
                    # 按'q'退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except RuntimeError as e:
                    print(f"获取帧时出错: {e}")
                    print("尝试重新初始化摄像头...")
                    self.pipeline.stop()
                    time.sleep(1)
                    self.pipeline.start(self.config)
                    
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 清理资源
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序结束")

# 使用示例
if __name__ == "__main__":
    # 你的模型路径
    model_path = "/home/rong/Pointnet_Pointnet2_pytorch/YOLO/runs/detect/tennis_track_train/weights/best.pt"
    
    try:
        # 创建追踪器并运行（先尝试不使用深度）
        tracker = RealSenseTennisTracker(model_path, use_depth=False)
        tracker.run()
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请检查摄像头连接")