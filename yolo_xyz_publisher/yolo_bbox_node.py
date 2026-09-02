#!/usr/bin/env python3

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"

import cv2
import numpy as np
import subprocess
import time

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

# autumn_node:
# 640x480 * 3개 = 1920x480
# 이후 0.5배 resize -> 960x240
STREAM_WIDTH = 640
STREAM_HEIGHT = 360
STREAM_FPS = 15

# H.264 bitrate
BITRATE = 700000

# EC2 MediaMTX
RTSP_URL = "rtsp://43.201.113.153:8554/yolo"

WINDOW_DETECTION = "YOLO Detection View (Combined 3-cam)"


class YoloBboxNode(Node):

    def __init__(self):
        super().__init__("yolo_bbox_node")

        self.bridge = CvBridge()

        # ====================================================
        # YOLO 모델
        # ====================================================

        self.get_logger().info(
            f"Loading YOLO model: {MODEL_PATH}"
        )

        self.model = YOLO(MODEL_PATH)

        # ====================================================
        # YOLO 문자열 결과 Publisher
        # spring/autumn node에서 ally, enemy, aruco 등 확인용
        # ====================================================

        self.string_publisher = self.create_publisher(
            String,
            "/yolo_bbox_raw",
            10
        )

        # ====================================================
        # 3-CAM 합성 영상 Subscriber
        # ====================================================

        self.color_sub = self.create_subscription(
            Image,
            "/ui_combined_vision",
            self.color_callback,
            1
        )

        # ====================================================
        # RTSP Stream
        # ====================================================

        self.stream_process = None
        self.stream_alive = False

        self.last_restart_time = 0.0

        self.start_streamer()

        self.get_logger().info(
            "YOLO BBOX Node initialized"
        )

        self.get_logger().info(
            "Subscribe: /ui_combined_vision"
        )

        self.get_logger().info(
            "Publish: /yolo_bbox_raw"
        )

        self.get_logger().info(
            f"RTSP: {RTSP_URL}"
        )

    # ========================================================
    # GStreamer RTSP 송출 시작
    # ========================================================

    def start_streamer(self):

        # 기존 프로세스가 남아 있으면 종료
        self.stop_streamer()

        self.get_logger().info(
            f"Starting YOLO RTSP stream -> {RTSP_URL}"
        )

        gst_command = [

            "gst-launch-1.0",
            "-q",

            # ------------------------------------------------
            # Python stdin에서 BGR raw frame 수신
            # ------------------------------------------------

            "fdsrc",
            "fd=0",
            "do-timestamp=true",

            "!",

            "rawvideoparse",
            "format=bgr",
            f"width={STREAM_WIDTH}",
            f"height={STREAM_HEIGHT}",
            f"framerate={STREAM_FPS}/1",

            "!",

            # ------------------------------------------------
            # 네트워크 느려졌을 때 옛날 frame 누적 방지
            # ------------------------------------------------

            "queue",
            "leaky=downstream",
            "max-size-buffers=1",

            "!",

            # ------------------------------------------------
            # OpenCV BGR -> Jetson encoder 입력
            # ------------------------------------------------

            "videoconvert",

            "!",

            "video/x-raw,format=BGRx",

            "!",

            "nvvidconv",

            "!",

            "video/x-raw(memory:NVMM),format=NV12",

            "!",

            # ------------------------------------------------
            # Jetson AGX Orin HW H.264 Encoder
            # ------------------------------------------------

            "nvv4l2h264enc",
            f"bitrate={BITRATE}",
            f"iframeinterval={STREAM_FPS}",
            "insert-sps-pps=true",
            "control-rate=1",
            "maxperf-enable=true",

            "!",

            "h264parse",
            "config-interval=-1",

            "!",

            # ------------------------------------------------
            # EC2 MediaMTX
            # ------------------------------------------------

            "rtspclientsink",
            f"location={RTSP_URL}",
            "protocols=tcp"
        ]

        try:

            self.stream_process = subprocess.Popen(
                gst_command,
                stdin=subprocess.PIPE,
                bufsize=0
            )

            self.stream_alive = True

            self.get_logger().info(
                "YOLO RTSP streamer started"
            )

        except Exception as e:

            self.get_logger().error(
                f"GStreamer start failed: {e}"
            )

            self.stream_process = None
            self.stream_alive = False

    # ========================================================
    # RTSP 종료
    # ========================================================

    def stop_streamer(self):

        if self.stream_process is None:
            return

        try:

            if self.stream_process.stdin is not None:
                self.stream_process.stdin.close()

        except Exception:
            pass

        try:

            if self.stream_process.poll() is None:

                self.stream_process.terminate()

                try:
                    self.stream_process.wait(
                        timeout=1.0
                    )

                except subprocess.TimeoutExpired:
                    self.stream_process.kill()

        except Exception:
            pass

        self.stream_process = None
        self.stream_alive = False

    # ========================================================
    # RTSP 죽었을 경우 재시작
    # ========================================================

    def check_streamer(self):

        if self.stream_process is not None:

            if self.stream_process.poll() is None:
                return True

        self.stream_alive = False

        now = time.time()

        # 2초마다만 재접속 시도
        if now - self.last_restart_time >= 2.0:

            self.last_restart_time = now

            self.get_logger().warning(
                "RTSP streamer disconnected -> restarting"
            )

            self.start_streamer()

        return self.stream_alive

    # ========================================================
    # YOLO Callback
    # ========================================================

    def color_callback(self, msg):

        try:

            # =================================================
            # ROS Image -> OpenCV
            # =================================================

            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            detection_image = image.copy()

            # =================================================
            # YOLO 추론
            # =================================================

            results = self.model(
                detection_image,
                verbose=False
            )

            # =================================================
            # Bounding Box
            # =================================================

            for result in results:

                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, cls_id, confidence in zip(
                    boxes,
                    classes,
                    confidences
                ):

                    x1, y1, x2, y2 = box.astype(int)

                    class_name = self.model.names[
                        int(cls_id)
                    ].lower()

                    # -----------------------------------------
                    # 클래스별 BOX 색상
                    # -----------------------------------------

                    if "enemy" in class_name:

                        box_color = (0, 0, 255)

                    elif "ally" in class_name:

                        box_color = (0, 255, 0)

                    elif "aruco" in class_name:

                        box_color = (255, 0, 255)

                    elif (
                        "robot_dog" in class_name
                        or "dog" in class_name
                    ):

                        box_color = (255, 255, 0)

                    elif (
                        "red" in class_name
                        and "light" in class_name
                    ):

                        box_color = (0, 0, 255)

                    elif (
                        "green" in class_name
                        and "light" in class_name
                    ):

                        box_color = (0, 255, 0)

                    else:

                        box_color = (0, 165, 255)

                    # -----------------------------------------
                    # Bounding Box
                    # -----------------------------------------

                    cv2.rectangle(
                        detection_image,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    # -----------------------------------------
                    # 클래스 + confidence
                    # -----------------------------------------

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        detection_image,
                        label,
                        (
                            x1,
                            max(y1 - 8, 15)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2
                    )

                    # -----------------------------------------
                    # 기존 String Publisher 유지
                    # -----------------------------------------

                    str_msg = String()

                    str_msg.data = class_name

                    self.string_publisher.publish(
                        str_msg
                    )

            # =================================================
            # RTSP 송출 크기 보정
            # =================================================

            if (
                detection_image.shape[1]
                != STREAM_WIDTH
                or
                detection_image.shape[0]
                != STREAM_HEIGHT
            ):

                stream_frame = cv2.resize(
                    detection_image,
                    (
                        STREAM_WIDTH,
                        STREAM_HEIGHT
                    ),
                    interpolation=cv2.INTER_AREA
                )

            else:

                stream_frame = detection_image

            # 메모리 연속성 보장
            stream_frame = np.ascontiguousarray(
                stream_frame,
                dtype=np.uint8
            )

            # =================================================
            # RTSP 송출
            # =================================================

            if self.check_streamer():

                try:

                    self.stream_process.stdin.write(
                        stream_frame.tobytes()
                    )

                except (
                    BrokenPipeError,
                    OSError
                ) as e:

                    self.get_logger().error(
                        f"YOLO RTSP pipe disconnected: {e}"
                    )

                    self.stream_alive = False

            # =================================================
            # Jetson 로컬 화면
            # =================================================

            cv2.imshow(
                WINDOW_DETECTION,
                detection_image
            )

            cv2.waitKey(1)

        except Exception as e:

            self.get_logger().error(
                f"YOLO callback error: {e}"
            )

    # ========================================================
    # 종료
    # ========================================================

    def destroy_node(self):

        self.get_logger().info(
            "Stopping YOLO BBOX Node"
        )

        self.stop_streamer()

        cv2.destroyAllWindows()

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = YoloBboxNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
