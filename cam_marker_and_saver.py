#!/usr/bin/env python3

import os
import json
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2

import tf2_ros
from tf2_ros import TransformException


class QRLandmarkSaver(Node):
    def __init__(self):
        super().__init__('qr_landmark_saver')

        self.bridge = CvBridge()
        self.detector = cv2.QRCodeDetector()

        self.image_topic = '/front_cam/image'
        self.min_qr_area_ratio = float(self.declare_parameter('min_qr_area_ratio', 0.022).value)
        self.min_qr_side_px = float(self.declare_parameter('min_qr_side_px', 160.0).value)
        self.reject_log_interval = max(1, int(self.declare_parameter('reject_log_interval', 10).value))
        self._reject_counter = 0

        self.map_frame = 'map'
        self.robot_frame = 'vehicle_blue/chassis'

        self.json_path = os.path.expanduser('/home/ibrahim/Desktop/AMR_project/qr_landmarks.json')

        self.landmarks = self.load_existing_landmarks()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info('QR Landmark Saver started')
        self.get_logger().info(f'Saving landmarks to: {self.json_path}')
        self.get_logger().info(f'Using transform: {self.map_frame} -> {self.robot_frame}')
        self.get_logger().info(
            f'QR near filter enabled: area_ratio>={self.min_qr_area_ratio:.4f}, '
            f'avg_side_px>={self.min_qr_side_px:.1f}'
        )

    def load_existing_landmarks(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_landmarks(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.landmarks, f, indent=4)

    def _normalize_points(self, points):
        if points is None:
            return None
        try:
            pts = points.reshape(-1, 2)
            if pts.shape[0] < 4:
                return None
            return pts
        except Exception:
            return None

    def _qr_size_metrics(self, points, image_shape):
        if points is None:
            return 0.0, 0.0
        h, w = image_shape[:2]
        if h <= 0 or w <= 0:
            return 0.0, 0.0

        area = abs(cv2.contourArea(points))
        area_ratio = area / float(w * h)

        side_sum = 0.0
        for idx in range(4):
            p1 = points[idx]
            p2 = points[(idx + 1) % 4]
            side_sum += math.hypot(float(p2[0] - p1[0]), float(p2[1] - p1[1]))
        avg_side = side_sum / 4.0
        return area_ratio, avg_side

    def _add_candidate(self, candidates, raw_text, points):
        qr_text = (raw_text or '').strip()
        if not qr_text:
            return
        normalized_points = self._normalize_points(points)
        pixel_area = abs(cv2.contourArea(normalized_points)) if normalized_points is not None else 0.0

        existing = candidates.get(qr_text)
        if existing is None or pixel_area > existing['pixel_area']:
            candidates[qr_text] = {
                'text': qr_text,
                'points': normalized_points,
                'pixel_area': pixel_area,
            }

    def _collect_candidates(self, image, candidates):
        # Multi QR pass
        try:
            ok, decoded_info, points, _ = self.detector.detectAndDecodeMulti(image)
            if ok and decoded_info:
                for idx, item in enumerate(decoded_info):
                    point_set = None
                    if points is not None and len(points) > idx:
                        point_set = points[idx]
                    self._add_candidate(candidates, item, point_set)
        except Exception:
            pass

        # Single QR pass
        try:
            data, points, _ = self.detector.detectAndDecode(image)
            self._add_candidate(candidates, data, points)
        except Exception:
            pass

    def decode_qr_candidates(self, cv_image):
        candidates = {}

        # Try raw frame first.
        self._collect_candidates(cv_image, candidates)

        # Fallback pass: thresholded image for harder QR conditions.
        if not candidates:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                2
            )
            self._collect_candidates(binary, candidates)

        return list(candidates.values())

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time()
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            yaw = self.quaternion_to_yaw(transform.transform.rotation)

            return x, y, z, yaw

        except TransformException as ex:
            self.get_logger().warn(f'Could not get robot pose: {ex}')
            return None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            candidates = self.decode_qr_candidates(cv_image)
            if not candidates:
                return

            pose = self.get_robot_pose()

            if pose is None:
                return

            x, y, z, yaw = pose
            saved_any = False

            for candidate in candidates:
                qr_text = candidate['text']
                if qr_text in self.landmarks:
                    continue

                area_ratio, avg_side = self._qr_size_metrics(candidate['points'], cv_image.shape)
                if area_ratio < self.min_qr_area_ratio or avg_side < self.min_qr_side_px:
                    self._reject_counter += 1
                    if self._reject_counter % self.reject_log_interval == 0:
                        self.get_logger().info(
                            f'QR "{qr_text}" rejected (too far): '
                            f'area_ratio={area_ratio:.4f}<{self.min_qr_area_ratio:.4f} or '
                            f'avg_side={avg_side:.1f}px<{self.min_qr_side_px:.1f}px'
                        )
                    continue

                self.landmarks[qr_text] = {
                    "frame": self.map_frame,
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": yaw
                }
                saved_any = True

                self.get_logger().info(
                    f'Saved QR landmark: {qr_text} at x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} '
                    f'(area_ratio={area_ratio:.4f}, avg_side={avg_side:.1f}px)'
                )

            if saved_any:
                self.save_landmarks()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = QRLandmarkSaver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
