#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class SaveOneCameraImage(Node):
    def __init__(self):
        super().__init__('save_one_camera_image')

        self.bridge = CvBridge()

        self.image_topic = '/front_cam/image'
        self.save_path = os.path.expanduser('~/test_camera_image.png')
        self.saved = False

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info('Waiting for image...')
        self.get_logger().info(f'Image topic: {self.image_topic}')
        self.get_logger().info(f'Will save to: {self.save_path}')

    def image_callback(self, msg):
        if self.saved:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            success = cv2.imwrite(self.save_path, cv_image)

            if success:
                self.saved = True
                self.get_logger().info(f'Saved image to: {self.save_path}')
                self.get_logger().info('You can open it with:')
                self.get_logger().info(f'xdg-open {self.save_path}')
            else:
                self.get_logger().error('cv2.imwrite failed')

        except Exception as e:
            self.get_logger().error(f'Error saving image: {e}')


def main(args=None):
    rclpy.init(args=args)

    node = SaveOneCameraImage()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()