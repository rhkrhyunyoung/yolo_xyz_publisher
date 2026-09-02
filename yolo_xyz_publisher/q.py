import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import serial  # 추가됨
import time    

import os
import subprocess


# ============================================================
# 기본 설정
# ============================================================

os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"

# q.py와 같은 폴더에 best.pt가 있다고 가정
MODEL_PATH = "/home/kudos/drokck_free/src/yolo_xyz_publisher/yolo_xyz_publisher/best.pt"

WINDOW_DETECTION = "YOLO Detection View"

# ------------------------------------------------------------
# RealSense 입력
# ------------------------------------------------------------

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ------------------------------------------------------------
# 브라우저 송출
# RealSense / YOLO 계산은 1280x720
# 송출만 640x360으로 축소
# ------------------------------------------------------------

STREAM_WIDTH = 640
STREAM_HEIGHT = 360
STREAM_FPS = 15

BITRATE = 700000

RTSP_URL = "rtsp://43.201.113.153:8554/yolo"

# DISPLAY가 있을 때만 로컬 OpenCV 창 표시
SHOW_LOCAL_WINDOW = bool(os.environ.get("DISPLAY"))


class YoloXyzNode(Node):

    def __init__(self):
        super().__init__("yolo_xyz_node")

        # ====================================================
        # OpenCV / YOLO
        # ====================================================

        self.bridge = CvBridge()

        self.get_logger().info(
            f"Loading YOLO model: {MODEL_PATH}"
        )

        self.model = YOLO(MODEL_PATH)

        # ====================================================
        # Camera Data
        # ====================================================

        self.intrinsics = None
        self.latest_depth_img = None

        # ====================================================
        # ROS2 Publisher
        # ====================================================

        self.string_publisher = self.create_publisher(
            String,
            "/yolo_detected_object",
            10
        )

        self.xyz_publisher = self.create_publisher(
            Vector3Stamped,
            "/yolo_object_xyz",
            10
        )

        # ====================================================
        # ROS2 Subscriber
        # ====================================================

        self.color_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.color_callback,
            1
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            1
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.info_callback,
            1
        )

        # ====================================================
        # Arduino Serial (LED 제어용)
        # ====================================================
        try:
            # 포트가 다를 경우 /dev/ttyUSB0 를 /dev/ttyACM0 등으로 수정하세요
            self.arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info("아두이노 연결 성공!")
        except Exception as e:
            self.get_logger().error(f"아두이노 연결 실패: {e}")
            self.arduino = None
            
        # ====================================================
        # RTSP Streaming
        # ====================================================

        self.yolo_stream = None
        self.stream_alive = False

        self.start_streamer()

        self.get_logger().info(
            "YOLO XYZ Node initialized"
        )

    def start_streamer(self):
        self.get_logger().info(f"Starting YOLO RTSP stream -> {RTSP_URL}")
        gst_command = [
            "gst-launch-1.0", "-e", "-v",
            "fdsrc", "fd=0", "do-timestamp=true", "!",
            "rawvideoparse", "format=bgr", f"width={STREAM_WIDTH}", f"height={STREAM_HEIGHT}", f"framerate={STREAM_FPS}/1", "!",
            "queue", "leaky=downstream", "max-size-buffers=1", "!",
            "videoconvert", "!",
            "video/x-raw,format=BGRx", "!",
            "nvvidconv", "!",
            "video/x-raw(memory:NVMM),format=NV12", "!",
            "nvv4l2h264enc", f"bitrate={BITRATE}", f"iframeinterval={STREAM_FPS}", "insert-sps-pps=true", "control-rate=1", "maxperf-enable=true", "!",
            "h264parse", "config-interval=-1", "!",
            "rtspclientsink", f"location={RTSP_URL}", "protocols=tcp"
        ]
        try:
            self.yolo_stream = subprocess.Popen(gst_command, stdin=subprocess.PIPE, bufsize=0)
            self.stream_alive = True
            self.get_logger().info("YOLO RTSP streamer started")
        except Exception as e:
            self.get_logger().error(f"Failed to start YOLO RTSP streamer: {e}")
            self.yolo_stream = None
            self.stream_alive = False

    def info_callback(self, msg):
        self.intrinsics = msg

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            self.get_logger().warning(f"Depth conversion error: {e}")

    def color_callback(self, msg):
        try:
            original_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"Color conversion error: {e}")
            return

        detection_image = original_image.copy()
        image_height, image_width = detection_image.shape[:2]

        try:
            results = self.model(detection_image, verbose=False)
        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")
            return

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, min(x1, image_width - 1))
                x2 = max(0, min(x2, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                y2 = max(0, min(y2, image_height - 1))
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                class_name = self.model.names[int(cls_id)].lower()

                # Bounding Box 색상 결정
                if "enemy" in class_name:
                    box_color = (0, 0, 255)
                elif "ally" in class_name:
                    box_color = (0, 255, 0)
                else:
                    box_color = (0, 165, 255)

                cv2.rectangle(detection_image, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(detection_image, f"[{class_name}]", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # ============================================
                # 시리얼 통신 및 문자열 처리 (수정됨)
                # ============================================
                sending_string = ""

                if "red" in class_name:
                    sending_string = "STOP"
                elif "green" in class_name:
                    sending_string = "GO"
                elif "enemy" in class_name:
                    sending_string = "enemy"
                    if self.arduino:
                        self.arduino.write(b'R') # 아두이노에 R(Red) 전송
                elif "ally" in class_name:
                    sending_string = "ally"
                    if self.arduino:
                        self.arduino.write(b'G') # 아두이노에 G(Green) 전송
                elif "supply_box" in class_name:
                    sending_string = "supply_box"
                elif "robot_dog" in class_name or "dog" in class_name:
                    sending_string = "robot_dog"

                if sending_string != "":
                    str_msg = String()
                    str_msg.data = sending_string
                    self.string_publisher.publish(str_msg)

                # XYZ 계산 (생략되지 않음)
                if (self.latest_depth_img is not None and self.intrinsics is not None and 
                    ("supply_box" in class_name or "robot_dog" in class_name or "aruco" in class_name)):
                    depth_height, depth_width = self.latest_depth_img.shape[:2]
                    if 0 <= cx < depth_width and 0 <= cy < depth_height:
                        depth_mm = self.latest_depth_img[cy, cx]
                        if depth_mm > 0:
                            depth_m = float(depth_mm) / 1000.0
                            fx, fy = self.intrinsics.k[0], self.intrinsics.k[4]
                            ppx, ppy = self.intrinsics.k[2], self.intrinsics.k[5]
                            if fx != 0 and fy != 0:
                                real_x = (cx - ppx) * depth_m / fx
                                real_y = (cy - ppy) * depth_m / fy
                                xyz_msg = Vector3Stamped()
                                xyz_msg.header.stamp = self.get_clock().now().to_msg()
                                xyz_msg.header.frame_id = "camera_link"
                                xyz_msg.vector.x, xyz_msg.vector.y, xyz_msg.vector.z = float(depth_m), float(-real_x), float(-real_y)
                                self.xyz_publisher.publish(xyz_msg)
                                cv2.circle(detection_image, (cx, cy), 6, (0, 255, 255), -1)
                                cv2.putText(detection_image, f"{depth_m:.2f}m", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if SHOW_LOCAL_WINDOW:
            try:
                cv2.imshow(WINDOW_DETECTION, detection_image)
                cv2.waitKey(1)
            except Exception: pass

        # RTSP 송출 부분
        if not self.stream_alive or self.yolo_stream is None: return
        if self.yolo_stream.poll() is not None:
            self.get_logger().error("GStreamer YOLO RTSP process exited")
            self.stream_alive = False
            return

        try:
            stream_image = cv2.resize(detection_image, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
            stream_image = np.ascontiguousarray(stream_image, dtype=np.uint8)
            self.yolo_stream.stdin.write(stream_image.tobytes())
        except (BrokenPipeError, OSError) as e:
            self.get_logger().error(f"YOLO RTSP pipe disconnected: {e}")
            self.stream_alive = False

    def destroy_node(self):
        if self.yolo_stream is not None:
            try: self.yolo_stream.stdin.close()
            except Exception: pass
            try:
                if self.yolo_stream.poll() is None:
                    self.yolo_stream.terminate()
                    self.yolo_stream.wait(timeout=2)
            except Exception:
                try: self.yolo_stream.kill()
                except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloXyzNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()
