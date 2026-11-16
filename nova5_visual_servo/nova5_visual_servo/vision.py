"""D435i color CameraInfo + QR corners -> timestamped target pose."""
import cv2
import numpy as np


def solve_qr(image, K, distortion, side, expected_text, max_error=2.):
    text, corners, _ = cv2.QRCodeDetector().detectAndDecode(image)
    if corners is None or not text or text != expected_text:
        return None
    s = side/2
    # QR object convention: x right, y down, z into marker at frontal view.
    points = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], np.float64)
    pixels = corners.reshape(4, 2).astype(np.float64)
    ok, rv, tv = cv2.solvePnP(points, pixels, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tv[2, 0] <= 0:
        return None
    projected, _ = cv2.projectPoints(points, rv, tv, K, distortion)
    error = np.sqrt(np.mean(np.sum((projected.reshape(4, 2)-pixels)**2, axis=1)))
    if error > max_error:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rv)[0]
    T[:3, 3] = tv.ravel()
    return T, error


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
    from scipy.spatial.transform import Rotation
    class Vision(Node):
        def __init__(self):
            super().__init__('qr_vision')
            self.declare_parameter('side_m', .1)
            self.declare_parameter('target_text', 'nova5-target')
            self.K = None
            self.bridge = CvBridge()
            self.pub = self.create_publisher(PoseStamped, '/visual_servo/target', 1)
            self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info, qos_profile_sensor_data)
            self.create_subscription(Image, '/camera/camera/color/image_raw', self.frame, qos_profile_sensor_data)
        def info(self, m):
            if m.distortion_model not in ('plumb_bob', 'rational_polynomial'):
                self.K = None
                return
            self.K, self.D = np.array(m.k).reshape(3, 3), np.array(m.d)
        def frame(self, m):
            if self.K is None:
                return
            result = solve_qr(self.bridge.imgmsg_to_cv2(m, 'bgr8'), self.K, self.D,
                              self.get_parameter('side_m').value, self.get_parameter('target_text').value)
            if result is None:
                return
            T, _ = result
            out = PoseStamped()
            out.header = m.header
            out.pose.position.x, out.pose.position.y, out.pose.position.z = T[:3, 3].tolist()
            quat = Rotation.from_matrix(T[:3, :3]).as_quat()
            out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = quat.tolist()
            self.pub.publish(out)
    rclpy.init()
    node = Vision()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
