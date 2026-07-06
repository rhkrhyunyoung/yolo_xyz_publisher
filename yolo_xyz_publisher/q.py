import cv2
import numpy as np
from ultralytics import YOLO
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import os

# --- Configuration Parameters ---
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
MODEL_PATH = '/home/rhkrgusdud/drokck_free/src/yolo_xyz_publisher/best.pt'
WINDOW_DETECTION = "2. YOLO Detection View (For Monitoring)"

class YoloXyzNode(Node):
    def __init__(self):
        super().__init__('yolo_xyz_node')
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
        
        # 카메라 내상수(Intrinsics) 저장을 위한 변수
        self.intrinsics = None
        
        # 퍼블리셔 설정
        self.string_publisher = self.create_publisher(String, '/yolo_detected_object', 10)
        self.xyz_publisher = self.create_publisher(Vector3Stamped, '/yolo_object_xyz', 10)

        # 드라이버에서 나오는 토픽 구독 (중요!)
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        self.latest_depth_img = None
        print(f"[INFO] YOLOv8 model loaded from {MODEL_PATH}")
        print("[INFO] YOLO XYZ Node initialized in Topic Subscriber mode.")

    def info_callback(self, msg):
        # 카메라 내상수 정보 업데이트 (3D 좌표 연산용)
        self.intrinsics = msg

    def depth_callback(self, msg):
        # 뎁스 영상을 OpenCV 형식으로 변환하여 저장
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def color_callback(self, msg):
        if self.latest_depth_img is None or self.intrinsics is None:
            return

        # 컬러 영상을 OpenCV 형식으로 변환
        original_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        detection_image = original_image.copy()

        # YOLO 추론
        results = self.model(detection_image, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                class_name = self.model.names[int(cls_id)].lower()

                # --- 시각화 색상 결정 (기존 로직 유지) ---
                if 'enemy' in class_name: box_color = (0, 0, 255)
                elif 'ally' in class_name: box_color = (0, 255, 0)
                elif 'aruco' in class_name: box_color = (255, 0, 255)
                elif 'robot_dog' in class_name or 'dog' in class_name: box_color = (255, 255, 0)
                elif 'red' in class_name and 'light' in class_name: box_color = (0, 0, 255)
                elif 'green' in class_name and 'light' in class_name: box_color = (0, 255, 0)
                else: box_color = (0, 165, 255)

                cv2.rectangle(detection_image, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(detection_image, f"[{class_name}]", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # --- 차체 C++ 전송용 문자열 매핑 ---
                sending_string = ""
                if 'red' in class_name: sending_string = "STOP"
                elif 'green' in class_name: sending_string = "GO"
                elif 'supply_box' in class_name: sending_string = "supply_box"
                elif 'robot_dog' in class_name or 'dog' in class_name: sending_string = "robot_dog"

                if sending_string != "":
                    str_msg = String()
                    str_msg.data = sending_string
                    self.string_publisher.publish(str_msg)

                # --- 3D 좌표 연산 (Z, -X, -Y 변환 포함) ---
                depth_mm = self.latest_depth_img[cy, cx]
                if depth_mm > 0:
                    depth_m = depth_mm / 1000.0
                    
                    # 핀홀 카메라 모델을 이용한 3D 좌표 복원
                    fx = self.intrinsics.k[0]
                    fy = self.intrinsics.k[4]
                    ppx = self.intrinsics.k[2]
                    ppy = self.intrinsics.k[5]

                    real_x = (cx - ppx) * depth_m / fx
                    real_y = (cy - ppy) * depth_m / fy
                    real_z = depth_m

                    xyz_msg = Vector3Stamped()
                    xyz_msg.header.stamp = self.get_clock().now().to_msg()
                    xyz_msg.header.frame_id = "camera_link"
                    xyz_msg.vector.x = float(real_z)   # 전방
                    xyz_msg.vector.y = float(-real_x)  # 좌측
                    xyz_msg.vector.z = float(-real_y)  # 상단

                    if 'supply_box' in class_name or 'robot_dog' in class_name or 'aruco' in class_name:
                        cv2.circle(detection_image, (cx, cy), 5, (0, 255, 255), -1)
                        self.xyz_publisher.publish(xyz_msg)

        cv2.imshow(WINDOW_DETECTION, detection_image)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = YoloXyzNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
