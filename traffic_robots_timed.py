#!/usr/bin/env python3

"""Simple timed traffic robots controller: Move 1min -> Rotate 180° -> Move 1min -> Repeat"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class TrafficRobotsTimed(Node):
    def __init__(self):
        super().__init__('traffic_robots_timed')
        
        # Parameters
        self.move_duration = float(self.declare_parameter('move_duration', 30.0).value)  # 60 seconds
        self.rotation_speed = float(self.declare_parameter('rotation_speed', 0.5).value)  # rad/s
        self.linear_speed = float(self.declare_parameter('linear_speed', 0.3).value)  # m/s
        self.update_rate = float(self.declare_parameter('update_rate', 10.0).value)
        
        # Calculate time to rotate 180 degrees (π radians)
        self.rotation_time = math.pi / self.rotation_speed  # Time to rotate 180°
        
        # Total cycle: move + rotate + move + rotate = 2*move + 2*rotate
        self.cycle_duration = 2 * self.move_duration + 2 * self.rotation_time
        
        # Publishers
        self.bot1_pub = self.create_publisher(Twist, '/traffic_bot_1/cmd_vel', 10)
        self.bot2_pub = self.create_publisher(Twist, '/traffic_bot_2/cmd_vel', 10)
        
        # Timer
        timer_period = 1.0 / self.update_rate
        self.timer = self.create_timer(timer_period, self.control_callback)
        self.start_time = None
        
        self.get_logger().info(
            f'Timed traffic controller started.\n'
            f'  Move duration: {self.move_duration}s\n'
            f'  Rotation time: {self.rotation_time:.2f}s (180° at {self.rotation_speed} rad/s)\n'
            f'  Linear speed: {self.linear_speed} m/s\n'
            f'  Cycle duration: {self.cycle_duration:.2f}s'
        )

    def control_callback(self):
        """Control both robots with timed pattern."""
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        
        # Get position in cycle (0 to cycle_duration)
        cycle_time = elapsed % self.cycle_duration
        
        # Determine state and command
        if cycle_time < self.move_duration:
            # Phase 1: Move forward
            state = "MOVING"
            bot1_cmd = self._make_twist(self.linear_speed, 0.0)
            bot2_cmd = self._make_twist(self.linear_speed, 0.0)
            
        elif cycle_time < self.move_duration + self.rotation_time:
            # Phase 2: Both rotate 180 degrees
            state = "ROTATING"
            bot1_cmd = self._make_twist(0.0, self.rotation_speed)
            bot2_cmd = self._make_twist(0.0, self.rotation_speed)
            
        elif cycle_time < 2 * self.move_duration + self.rotation_time:
            # Phase 3: Move forward (after rotating)
            state = "MOVING (rotated)"
            bot1_cmd = self._make_twist(self.linear_speed, 0.0)
            bot2_cmd = self._make_twist(self.linear_speed, 0.0)
            
        else:
            # Phase 4: Both rotate 180 degrees back
            state = "ROTATING (back)"
            bot1_cmd = self._make_twist(0.0, self.rotation_speed)
            bot2_cmd = self._make_twist(0.0, self.rotation_speed)
        
        # Publish commands
        self.bot1_pub.publish(bot1_cmd)
        self.bot2_pub.publish(bot2_cmd)
        
        # Log every 5 seconds
        if int(elapsed) % 5 == 0 and elapsed != int(elapsed - 0.1):
            self.get_logger().info(
                f'Elapsed: {elapsed:.0f}s | Cycle time: {cycle_time:.1f}s/{self.cycle_duration:.1f}s | State: {state}'
            )

    def _make_twist(self, linear_x: float, angular_z: float) -> Twist:
        """Create a Twist message."""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        return msg

    def stop_all(self):
        """Stop both robots."""
        stop = Twist()
        self.bot1_pub.publish(stop)
        self.bot2_pub.publish(stop)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficRobotsTimed()
    
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
