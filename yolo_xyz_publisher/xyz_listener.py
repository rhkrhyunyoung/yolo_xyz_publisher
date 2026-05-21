import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class XYZListener(Node):
    def __init__(self):
        super().__init__('xyz_listener')
        self.subscription = self.create_subscription(
            Point,
            'object_position',
            self.listener_callback,
            10
        )
        self.get_logger().info('Subscribed to /object_position')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received XYZ: X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f} (m)')

# ✅ 반드시 필요
def main(args=None):
    rclpy.init(args=args)
    node = XYZListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
