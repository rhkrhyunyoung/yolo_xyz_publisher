from setuptools import find_packages, setup
import os

package_name = 'yolo_xyz_publisher'

# weight 파일이 있는지 확인
weights_path = 'best.pt'
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

# weights 포함되면 추가
if os.path.exists(weights_path):
    data_files.append(
        (os.path.join('share', package_name, 'weights'), [weights_path])
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rim',
    maintainer_email='9y0r1m@gmail.com',
    description='YOLOv8 + RealSense depth 추출 및 XYZ 좌표 출력 ROS 2 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'xyz_listener = yolo_xyz_publisher.xyz_listener:main',
            'q = yolo_xyz_publisher.q:main',
        ],
    },
)
