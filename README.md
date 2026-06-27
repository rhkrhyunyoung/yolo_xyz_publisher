# yolo_xyz Spatial Object Publisher
Real-time 3D Coordinate Estimation via YOLO Detection & Depth Fusion

![alt text](https://img.shields.io/badge/ROS2-Humble-0A0FF9?style=for-the-badge&logo=ros&logoColor=white)


![alt text](https://img.shields.io/badge/YOLO-v8/v11-00FFFFFF?style=for-the-badge&logo=yolo&logoColor=white)


![alt text](https://img.shields.io/badge/python-3.10+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

"Bridging the gap between 2D Computer Vision and 3D Robotics."
The project combines YOLO's strong object recognition capabilities with the Depth Camera's distance data, converting the pixel coordinates on the image into three-dimensional space coordinates (XYZ) in the real world and transmitting them in real time to ROS 2 topics.

# Key Technical Features
1. 2D-to-3D Projection (Pinhole Camera Model)
단순히 중앙값의 깊이를 읽는 것을 넘어, 카메라의 내적 파라미터(Intrinsic Parameters:fx,fy,cx,cy​)를 활용하여 픽셀 좌표를 실제 미터(m) 단위의 공간 좌표로 투영합니다.
- Precision: 객체의 중앙 바운딩 박스(Bounding Box)를 기준으로 가장 안정적인 깊이 데이터를 추출합니다.

2. ROS 2 Real-time Messaging
추출된 XYZ 좌표를 전용 메시지 타입(geometry_msgs/Point 또는 커스텀 메시지)으로 /detected_object/xyz 토픽에 퍼블리싱합니다.
- Latency-Optimized: YOLO 추론과 뎁스 정렬(Depth Alignment) 과정을 최적화하여 실시간 로봇 제어에 적합한 프레임워크를 제공합니다.

3. Spatial Centroid Calculation
객체의 바운딩 박스 내 노이즈를 최소화하기 위해 중앙 영역의 깊이 평균값을 계산하거나 정렬된 뎁스 맵(Aligned Depth Map)을 사용하여 정확도를 높였습니다.

# Project Structure
```
yolo_xyz_publisher/
├── yolo_xyz_publisher/
│   ├── main_publisher.py      # Core ROS 2 Node
│   ├── detection_logic.py     # YOLO inference & BBox processing
│   └── transformation.py      # Pixel to 3D Space Projection logic
├── launch/
│   └── yolo_xyz.launch.py     # Multi-node execution launch file
└── config/
    └── params.yaml            # Camera intrinsics & YOLO settings
```

# Getting Started
```
ros2 launch yolo_xyz_publisher yolo_xyz.launch.py
```

# Evaluation & Visualization
- RViz2 Support: 실시간으로 퍼블리싱되는 XYZ 데이터를 RViz2 상의 Marker 또는 TF로 시각화하여 로봇의 실제 인식 위치를 확인할 수 있습니다.
- Terminal Feedback: 매 프레임 검출된 물체의 클래스명과 좌표 정보를 출력합니다.
 ex) [INFO] Detected: [Bottle] at X: 0.23m, Y: -0.05m, Z: 1.12m
