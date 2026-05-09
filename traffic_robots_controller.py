#!/usr/bin/env python3

"""Drive two traffic robots back and forth to create dynamic obstacles."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TrafficRobotsController(Node):
    def __init__(self):
        super().__init__('traffic_robots_controller')

        self.bot1_topic = str(self.declare_parameter('bot1_topic', '/traffic_bot_1/cmd_vel').value)
        self.bot2_topic = str(self.declare_parameter('bot2_topic', '/traffic_bot_2/cmd_vel').value)
        self.bot1_speed = float(self.declare_parameter('bot1_speed', 0.45).value)
        self.bot2_speed = float(self.declare_parameter('bot2_speed', 0.35).value)
        self.cycle_period = float(self.declare_parameter('cycle_period', 6.0).value)
        self.publish_rate = float(self.declare_parameter('publish_rate', 10.0).value)

        self.bot1_pub = self.create_publisher(Twist, self.bot1_topic, 10)
        self.bot2_pub = self.create_publisher(Twist, self.bot2_topic, 10)

        self.start_time = self.get_clock().now()
        self.last_phase = None

        timer_period = 1.0 / max(self.publish_rate, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Traffic robots controller started')
        self.get_logger().info(
            f'bot1_topic={self.bot1_topic}, bot2_topic={self.bot2_topic}, '
            f'bot1_speed={self.bot1_speed:.2f}, bot2_speed={self.bot2_speed:.2f}, '
            f'cycle_period={self.cycle_period:.2f}s'
        )

    def _make_twist(self, linear_x: float) -> Twist:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = 0.0
        return msg

    def timer_callback(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        phase = 1.0 if int(elapsed / max(self.cycle_period, 0.1)) % 2 == 0 else -1.0

        # Move in opposite directions for better traffic interaction.
        bot1_cmd = self._make_twist(phase * self.bot1_speed)
        bot2_cmd = self._make_twist(-phase * self.bot2_speed)

        self.bot1_pub.publish(bot1_cmd)
        self.bot2_pub.publish(bot2_cmd)

        if self.last_phase is None or phase != self.last_phase:
            direction = 'forward' if phase > 0 else 'backward'
            self.get_logger().info(f'Switched direction: {direction}')
            self.last_phase = phase

    def stop_all(self):
        stop = Twist()
        self.bot1_pub.publish(stop)
        self.bot2_pub.publish(stop)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficRobotsController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
