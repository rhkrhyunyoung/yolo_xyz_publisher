#!/usr/bin/env python3

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"

import cv2
import numpy as np
import subprocess
import time
import serial

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO


# ============================================================
# 설정
# ============================================================

MODEL_PATH = (
    "/home/kudos/drokck_free/src/"
    "yolo_xyz_publisher/yolo_xyz_publisher/best.pt"
)

STREAM_WIDTH = 640
STREAM_HEIGHT = 360
STREAM_FPS = 15
BITRATE = 700000
RTSP_URL = "rtsp://43.201.113.153:8554/yolo"
WINDOW_DETECTION = "YOLO Detection View (Combined 3-cam)"

# --- 추가 설정 ---
LED_HOLD_TIME = 2.0  # 물체가 사라져도 불빛을 유지할 시간 (초)
# ----------------


class YoloBboxNode(Node):

    def __init__(self):
        super().__init__("yolo_bbox_node")

        self.bridge = CvBridge()

        # ====================================================
        # YOLO 모델
        # ====================================================
        self.get_logger().info(f"Loading YOLO model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

        # ====================================================
        # Arduino Serial 설정
        # ====================================================
        try:
            self.arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info("아두이노 연결 성공! (3-CAM 노드)")
        except Exception as e:
            self.get_logger().error(f"아두이노 연결 실패: {e}")
            self.arduino = None

        # --- 상태 유지를 위한 변수 추가 ---
        self.current_led_state = "O"    # 현재 LED 상태 (R, G, O)
        self.last_detection_time = 0.0  # 마지막으로 물체를 발견한 시간
        self.last_sent_signal = ""      # 아두이노로 마지막에 실제 전송한 신호
        self.last_arduino_send_time = 0.0
        # -------------------------------

        # ====================================================
        # YOLO 결과 Publisher
        # ====================================================
        self.string_publisher = self.create_publisher(String, "/yolo_bbox_raw", 10)

        # ====================================================
        # 3-CAM 합성 영상 Subscriber
        # ====================================================
        self.color_sub = self.create_subscription(
            Image, "/ui_combined_vision", self.color_callback, 1
        )

        # ====================================================
        # RTSP Stream
        # ====================================================
        self.stream_process = None
        self.stream_alive = False
        self.last_restart_time = 0.0
        self.start_streamer()

        self.get_logger().info("YOLO BBOX Node initialized (with LED Hold)")

    def start_streamer(self):
        self.stop_streamer()
        gst_command = [
            "gst-launch-1.0", "-q", "fdsrc", "fd=0", "do-timestamp=true", "!",
            "rawvideoparse", "format=bgr", f"width={STREAM_WIDTH}", f"height={STREAM_HEIGHT}", f"framerate={STREAM_FPS}/1", "!",
            "queue", "leaky=downstream", "max-size-buffers=1", "!",
            "videoconvert", "!", "video/x-raw,format=BGRx", "!",
            "nvvidconv", "!", "video/x-raw(memory:NVMM),format=NV12", "!",
            "nvv4l2h264enc", f"bitrate={BITRATE}", f"iframeinterval={STREAM_FPS}", "insert-sps-pps=true", "control-rate=1", "maxperf-enable=true", "!",
            "h264parse", "config-interval=-1", "!",
            "rtspclientsink", f"location={RTSP_URL}", "protocols=tcp"
        ]
        try:
            self.stream_process = subprocess.Popen(gst_command, stdin=subprocess.PIPE, bufsize=0)
            self.stream_alive = True
        except Exception as e:
            self.get_logger().error(f"GStreamer start failed: {e}")

    def stop_streamer(self):
        if self.stream_process:
            try: self.stream_process.stdin.close()
            except: pass
            try: self.stream_process.terminate()
            except: pass
        self.stream_process = None
        self.stream_alive = False

    def check_streamer(self):
        if self.stream_process and self.stream_process.poll() is None: return True
        self.stream_alive = False
        now = time.time()
        if now - self.last_restart_time >= 2.0:
            self.last_restart_time = now
            self.start_streamer()
        return self.stream_alive

    def color_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            detection_image = image.copy()
            results = self.model(detection_image, verbose=False)

            now = time.time()
            enemy_found = False
            ally_found = False

            for result in results:
                if result.boxes is None: continue
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, cls_id, confidence in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = self.model.names[int(cls_id)].lower()

                    # 박스 색상 설정 및 발견 상태 업데이트
                    if "enemy" in class_name: 
                        box_color = (0, 0, 255)
                        enemy_found = True
                    elif "ally" in class_name: 
                        box_color = (0, 255, 0)
                        ally_found = True
                    else: 
                        box_color = (0, 165, 255)

                    cv2.rectangle(detection_image, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(detection_image, f"{class_name} {confidence:.2f}", 
                                (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # ROS Topic 전송 (문자열)
                    sending_string = ""
                    if "enemy" in class_name: sending_string = "enemy"
                    elif "ally" in class_name: sending_string = "ally"
                    elif "red" in class_name: sending_string = "STOP"
                    elif "green" in class_name: sending_string = "GO"
                    elif "supply_box" in class_name: sending_string = "supply_box"
                    elif "robot_dog" in class_name or "dog" in class_name: sending_string = "robot_dog"

                    if sending_string != "":
                        str_msg = String()
                        str_msg.data = sending_string
                        self.string_publisher.publish(str_msg)

            # ====================================================
            # LED 상태 결정 로직 (상태 유지 기능 포함)
            # ====================================================
            if enemy_found:
                self.current_led_state = "R"
                self.last_detection_time = now
            elif ally_found:
                # 적군이 없을 때만 아군 상태로 변경
                self.current_led_state = "G"
                self.last_detection_time = now
            else:
                # 아무것도 발견되지 않았을 때
                # 마지막 발견 후 LED_HOLD_TIME(2초)이 지났을 때만 꺼짐('O')으로 변경
                if now - self.last_detection_time > LED_HOLD_TIME:
                    self.current_led_state = "O"

            # ====================================================
            # 아두이노 데이터 전송 (부하 방지)
            # ====================================================
            if self.arduino:
                # 상태가 바뀌었거나, 0.5초마다 강제로 상태 갱신
                if (self.current_led_state != self.last_sent_signal) or (now - self.last_arduino_send_time > 0.5):
                    try:
                        self.arduino.write(self.current_led_state.encode())
                        self.last_sent_signal = self.current_led_state
                        self.last_arduino_send_time = now
                        # self.get_logger().info(f"Send to Arduino: {self.current_led_state}")
                    except Exception as e:
                        self.get_logger().error(f"Serial Write Error: {e}")

            # RTSP 송출 및 화면 표시
            if self.check_streamer():
                stream_frame = cv2.resize(detection_image, (STREAM_WIDTH, STREAM_HEIGHT))
                self.stream_process.stdin.write(np.ascontiguousarray(stream_frame).tobytes())

            cv2.imshow(WINDOW_DETECTION, detection_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"YOLO callback error: {e}")

    def destroy_node(self):
        self.stop_streamer()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloBboxNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()
