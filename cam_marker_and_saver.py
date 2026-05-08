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

        self.map_frame = 'map'
        self.robot_frame = 'vehicle_blue/chassis'

        self.json_path = os.path.expanduser('~/qr_landmarks.json')

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

            data, points, _ = self.detector.detectAndDecode(cv_image)

            if not data:
                return

            qr_text = data.strip()

            if qr_text in self.landmarks:
                return

            pose = self.get_robot_pose()

            if pose is None:
                return

            x, y, z, yaw = pose

            self.landmarks[qr_text] = {
                "frame": self.map_frame,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw
            }

            self.save_landmarks()

            self.get_logger().info(
                f'Saved QR landmark: {qr_text} at x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}'
            )

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