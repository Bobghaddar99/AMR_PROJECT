#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class FrontierGenerator(Node):
    def __init__(self):
        super().__init__('frontier_generator')

        self.declare_parameter('auto_map', 1)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            '/frontier_markers',
            10
        )

        self.cost_map = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None

        self.marker_id = 0

        self.get_logger().info("Frontier Generator node started.")

    def map_callback(self, msg):
        self.cost_map = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width)
        )

        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.generate_frontiers()

    def generate_frontiers(self):
        if self.cost_map is None:
            return

        height, width = self.cost_map.shape
        frontiers = []

        for y in range(height):
            for x in range(width):
                if self.cost_map[y, x] == 0:
                    if self.has_unknown_neighbors(x, y):
                        frontiers.append((x, y))

        self.publish_markers(frontiers)

    def has_unknown_neighbors(self, x, y):
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy

            if 0 <= nx < self.cost_map.shape[1] and 0 <= ny < self.cost_map.shape[0]:
                if self.cost_map[ny, nx] == -1:
                    return True

        return False

    def publish_markers(self, frontiers):
        automap = self.get_parameter('auto_map').value

        if automap != 1:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontiers"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for fx, fy in frontiers:
            point = Point()
            point.x = (fx + 0.5) * self.resolution + self.origin_x
            point.y = (fy + 0.5) * self.resolution + self.origin_y
            point.z = 0.0
            marker.points.append(point)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)

    node = FrontierGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()