#!/usr/bin/env python3

import time
import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32

from tf2_ros import Buffer, TransformListener


status = None


class FrontierWeightedCalculator(Node):
    def __init__(self):
        super().__init__('frontier_weighted_calculator')

        self.declare_parameter('size_weight', 1.0)
        self.declare_parameter('distance_weight', 0.3)
        self.declare_parameter('min_size', 3)
        self.declare_parameter('stuck_time', 40.0)
        self.declare_parameter('robot_size', 2.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'vehicle_blue/chassis')
        self.declare_parameter('cluster_eps', 0.3)
        self.declare_parameter('cluster_min_samples', 1)

        self.size_weight = self.get_parameter('size_weight').value
        self.distance_weight = self.get_parameter('distance_weight').value
        self.min_size = self.get_parameter('min_size').value
        self.robot_stuck_timeout = self.get_parameter('stuck_time').value
        self.robot_size = self.get_parameter('robot_size').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.cluster_eps = self.get_parameter('cluster_eps').value
        self.cluster_min_samples = self.get_parameter('cluster_min_samples').value

        self.frontier_sub = self.create_subscription(
            Marker,
            '/frontier_markers',
            self.frontier_callback,
            10
        )

        self.mean_marker_pub = self.create_publisher(
            Marker,
            '/mean_frontier_markers',
            10
        )

        self.number_marker_pub = self.create_publisher(
            Marker,
            '/frontier_number_markers',
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.nav_goal_pub = self.create_publisher(
            PoseStamped,
            '/goal',
            10
        )

        self.distance_pub = self.create_publisher(
            Float32,
            '/frontier_distance',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.frontier_points = np.empty((0, 2))
        self.last_published_goal = None

        self.robot_positions = []
        self.stuck_start_time = None
        self.initial_position = None

        self.prev_number_count = 0

        self.timer = self.create_timer(0.5, self.timer_callback)
        self.initial_pose_timer = self.create_timer(1.0, self.save_initial_position_once)

        self.get_logger().info("Frontier weighted calculator started.")

    def save_initial_position_once(self):
        if self.initial_position is not None:
            return

        self.initial_position = self.get_robot_position()

        if self.initial_position is not None:
            self.get_logger().info(
                f"Initial position saved: {self.initial_position}"
            )

            self.initial_pose_timer.cancel()

    def frontier_callback(self, msg):
        self.get_logger().info(f"Frontier callback received, action={msg.action}, points={len(msg.points)}")
        
        if msg.action == Marker.DELETEALL or msg.action == Marker.DELETE:
            self.get_logger().info("Delete action received")
            return

        if not msg.points:
            self.get_logger().warn("No points in marker message")
            return

        self.frontier_points = np.array([
            [p.x, p.y] for p in msg.points
        ])
        
        self.get_logger().info(f"Updated frontier_points, shape: {self.frontier_points.shape}")

    def timer_callback(self):
        if len(self.frontier_points) == 0:
            return

        frontier_groups = self.group_frontiers(self.frontier_points)

        if not frontier_groups:
            return

        robot_position = self.get_robot_position()

        if robot_position is None:
            self.get_logger().warn("Could not get robot position.")
            return

        self.publish_mean_markers(frontier_groups)
        self.publish_number_markers(frontier_groups)

        self.check_robot_stuck(robot_position)

        if self.robot_is_stuck():
            self.get_logger().warn("Robot is stuck, navigating to initial position.")
            self.navigate_to_initial_position()
        else:
            self.publish_weighted_nav_goal(frontier_groups, robot_position)

    def group_frontiers(self, frontier_points):
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples
        ).fit(frontier_points)

        labels = clustering.labels_
        grouped_points = []

        for label in set(labels):
            if label == -1:
                continue

            group = frontier_points[labels == label]

            if len(group) >= self.min_size:
                grouped_points.append(group)

        return grouped_points

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time()
            )

            trans = transform.transform.translation
            return np.array([trans.x, trans.y])

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def publish_mean_markers(self, frontier_groups):
        mean_marker = Marker()
        mean_marker.header.frame_id = self.map_frame
        mean_marker.header.stamp = self.get_clock().now().to_msg()
        mean_marker.ns = "mean_frontier"
        mean_marker.id = 0
        mean_marker.type = Marker.SPHERE_LIST
        mean_marker.action = Marker.ADD
        mean_marker.pose.orientation.w = 1.0

        mean_marker.scale.x = 0.3
        mean_marker.scale.y = 0.3
        mean_marker.scale.z = 0.3

        mean_marker.color.a = 1.0
        mean_marker.color.r = 1.0
        mean_marker.color.g = 1.0
        mean_marker.color.b = 0.0

        for group in frontier_groups:
            mean_x = np.mean(group[:, 0])
            mean_y = np.mean(group[:, 1])

            point = Point()
            point.x = float(mean_x)
            point.y = float(mean_y)
            point.z = 0.0

            mean_marker.points.append(point)

        self.mean_marker_pub.publish(mean_marker)

    def publish_number_markers(self, frontier_groups):
        current_count = len(frontier_groups)

        for idx, group in enumerate(frontier_groups):
            mean_x = np.mean(group[:, 0])
            mean_y = np.mean(group[:, 1])

            number_marker = Marker()
            number_marker.header.frame_id = self.map_frame
            number_marker.header.stamp = self.get_clock().now().to_msg()
            number_marker.ns = "frontier_numbers"
            number_marker.id = idx
            number_marker.type = Marker.TEXT_VIEW_FACING
            number_marker.action = Marker.ADD

            number_marker.pose.position.x = float(mean_x)
            number_marker.pose.position.y = float(mean_y)
            number_marker.pose.position.z = 0.2
            number_marker.pose.orientation.w = 1.0

            number_marker.scale.z = 0.5

            number_marker.color.a = 1.0
            number_marker.color.r = 1.0
            number_marker.color.g = 1.0
            number_marker.color.b = 1.0

            number_marker.text = str(idx)

            self.number_marker_pub.publish(number_marker)

        for old_id in range(current_count, self.prev_number_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = self.map_frame
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "frontier_numbers"
            delete_marker.id = old_id
            delete_marker.action = Marker.DELETE

            self.number_marker_pub.publish(delete_marker)

        self.prev_number_count = current_count

    def check_robot_stuck(self, robot_position):
        if self.initial_position is None:
            return

        self.robot_positions.append(robot_position)

        if len(self.robot_positions) > 10:
            self.robot_positions.pop(0)

    def robot_is_stuck(self):
        if len(self.robot_positions) < 2:
            return False

        recent_positions = np.array(self.robot_positions)

        min_x, min_y = np.min(recent_positions, axis=0)
        max_x, max_y = np.max(recent_positions, axis=0)

        box_size = max(max_x - min_x, max_y - min_y)

        if box_size <= 2.0 * self.robot_size:
            if self.stuck_start_time is None:
                self.stuck_start_time = time.time()

            elif time.time() - self.stuck_start_time > self.robot_stuck_timeout:
                self.stuck_start_time = None
                return True
        else:
            self.stuck_start_time = None

        return False

    def navigate_to_initial_position(self):
        global status

        if self.initial_position is None:
            return

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.map_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = float(self.initial_position[0])
        goal_pose.pose.position.y = float(self.initial_position[1])
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        status = False

        self.goal_pub.publish(goal_pose)
        self.nav_goal_pub.publish(goal_pose)

        self.last_published_goal = (
            goal_pose.pose.position.x,
            goal_pose.pose.position.y
        )

        robot_position = self.get_robot_position()

        if robot_position is not None:
            dx = robot_position[0] - self.initial_position[0]
            dy = robot_position[1] - self.initial_position[1]

            distance = max(abs(dx), abs(dy))

            distance_msg = Float32()
            distance_msg.data = float(distance)

            self.distance_pub.publish(distance_msg)

    def publish_weighted_nav_goal(self, frontier_groups, robot_position):
        global status

        best_goal = None
        best_weight = float('-inf')
        best_distance = 0.0

        for group in frontier_groups:
            mean_x = np.mean(group[:, 0])
            mean_y = np.mean(group[:, 1])

            dx = mean_x - robot_position[0]
            dy = mean_y - robot_position[1]

            distance_to_robot = max(abs(dx), abs(dy))

            weight = (
                len(group) * self.size_weight
                - distance_to_robot * self.distance_weight
            )

            if weight > best_weight:
                best_weight = weight
                best_goal = (float(mean_x), float(mean_y))
                best_distance = float(distance_to_robot)

        if best_goal is None:
            return

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.map_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = best_goal[0]
        goal_pose.pose.position.y = best_goal[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        distance_msg = Float32()
        distance_msg.data = best_distance
        self.distance_pub.publish(distance_msg)

        if self.last_published_goal != best_goal:
            self.goal_pub.publish(goal_pose)
            self.nav_goal_pub.publish(goal_pose)
            self.last_published_goal = best_goal
            status = None

            self.get_logger().info(
                f"Published goal: {best_goal}, weight={best_weight}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = FrontierWeightedCalculator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()