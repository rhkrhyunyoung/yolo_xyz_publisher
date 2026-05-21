import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import os

# --- Configuration Parameters ---
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
MODEL_PATH = '/home/rhkrgusdud/yolo_xyz_publisher/best.pt'
COLOR_WIDTH, COLOR_HEIGHT, COLOR_FORMAT, COLOR_FPS = 640, 480, rs.format.bgr8, 30
DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FORMAT, DEPTH_FPS = 640, 480, rs.format.z16, 30
WINDOW_NAME = "YOLO + Depth & XYZ"

def initialize_realsense_pipeline(config):
    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(config)
        print("[INFO] RealSense pipeline started successfully.")
        return pipeline, profile
    except Exception as e:
        print(f"[ERROR] Failed to start RealSense pipeline: {e}", file=sys.stderr)
        return None, None

def get_depth_scale_and_intrinsics(profile):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale: {depth_scale:.4f}")
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    print(f"[INFO] Color camera intrinsics: {color_intrinsics}")
    return depth_scale, color_intrinsics

def process_frames(pipeline, align):
    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("[DEBUG] Frames not ready, skipping iteration.")
            return None, None
        return color_frame, depth_frame
    except RuntimeError as e:
        print(f"[ERROR] Error processing frames: {e}", file=sys.stderr)
        return None, None

def main():
    rclpy.init()
    node = rclpy.create_node('yolo_xyz_node')
    publisher = node.create_publisher(Point, 'object_position', 10)

    try:
        model = YOLO(MODEL_PATH)
        print(f"[INFO] YOLOv8 model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv8 model: {e}", file=sys.stderr)
        rclpy.shutdown()
        return

    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FORMAT, DEPTH_FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, COLOR_FORMAT, COLOR_FPS)

    pipeline, profile = initialize_realsense_pipeline(config)
    if pipeline is None:
        rclpy.shutdown()
        return

    depth_scale, color_intrinsics = get_depth_scale_and_intrinsics(profile)
    align = rs.align(rs.stream.color)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, 100, 100)

    pixel_scale = 0.5  # mm per pixel (approx.)

    try:
        while True:
            color_frame, depth_frame = process_frames(pipeline, align)
            if color_frame is None or depth_frame is None:
                continue

            color_image = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape((COLOR_HEIGHT, COLOR_WIDTH, 3))

            if color_image.ndim != 3 or color_image.shape[2] != 3:
                print(f"[ERROR] Invalid image shape: {color_image.shape}")
                continue

            try:
                results = model(color_image, verbose=False)
            except Exception as e:
                print(f"[ERROR] YOLO inference failed: {e}", file=sys.stderr)
                continue

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if 0 <= cx < DEPTH_WIDTH and 0 <= cy < DEPTH_HEIGHT:
                        depth_m = depth_frame.get_distance(cx, cy)

                        if depth_m > 0:
                            point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], depth_m)
                            real_x, real_y, real_z = point_3d

                            # 시각화 및 정보 출력
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(color_image, f"Robot XYZ: {real_x:.2f}, {-real_y:.2f}, {real_z:.2f} m", 
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            cv2.circle(color_image, (cx, cy), 3, (0, 0, 255), -1)

                            # ROS 2 메시지 발행
                            msg = Point()
                            msg.x = float(real_z)   # 카메라의 정면(Z)이 로봇의 앞방향(X)
                            msg.y = float(real_x)  # 카메라의 오른쪽(X)이 로봇의 왼쪽(Y)
                            msg.z = float(-real_y)  # 카메라의 아래쪽(Y)에 -를 붙여 로봇의 위쪽(Z)
                            publisher.publish(msg)
                            
                            print(f"[INFO] Robot XYZ (m): X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f}")
                        else:
                            print("[WARN] Depth is 0, skipping...")
                    else:
                        print(f"[WARN] Invalid point: ({cx}, {cy})")

            cv2.imshow(WINDOW_NAME, color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit signal received.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}", file=sys.stderr)
    finally:
        print("[INFO] Cleaning up...")
        if pipeline:
            pipeline.stop()
            print("[INFO] RealSense pipeline stopped.")
        if rclpy.ok():
            rclpy.shutdown()
            print("[INFO] ROS 2 shutdown complete.")
        cv2.destroyAllWindows()
        print("[INFO] OpenCV windows closed.")

if __name__ == "__main__":
    main()
