"""
Audio alert node — run on the ground station (not the drone).

Subscribes to /pressure_analysis and plays distinct audio when the
manometer reading is above or below the competition limit.

Audio files (MP3) are configured via ROS2 parameters:
  audio_above_limit: /path/to/above.mp3   (played when pressure > limit)
  audio_below_limit: /path/to/below.mp3   (played when pressure <= limit)

Requires 'mpg123' to be installed:
  sudo apt install mpg123

If the files are not found, falls back to espeak-ng TTS (requires
  sudo apt install espeak-ng).

Run on the ground station:
  source /opt/ros/humble/setup.bash
  ros2 run manometro_detector audio_alert_node \
    --ros-args -p audio_above_limit:=/path/above.mp3 \
               -p audio_below_limit:=/path/below.mp3

Make sure ROS_DOMAIN_ID matches the drone's value.
"""

import subprocess
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class AudioAlertNode(Node):
    def __init__(self):
        super().__init__('audio_alert_node')

        self.declare_parameter('audio_above_limit', '')
        self.declare_parameter('audio_below_limit', '')

        self._audio_above = self.get_parameter('audio_above_limit').value
        self._audio_below = self.get_parameter('audio_below_limit').value

        if self._audio_above:
            self.get_logger().info(f'Audio ACIMA : {self._audio_above}')
        if self._audio_below:
            self.get_logger().info(f'Audio ABAIXO: {self._audio_below}')
        if not self._audio_above and not self._audio_below:
            self.get_logger().info('Sem arquivos de audio configurados — usando espeak-ng TTS.')

        self.create_subscription(
            String, '/pressure_analysis', self._callback, 10)

        self.get_logger().info('Audio Alert Node pronto. Aguardando leituras...')

    def _callback(self, msg: String):
        if not msg.data:
            return

        is_above = 'above' in msg.data.lower()
        label    = 'ACIMA DO LIMITE' if is_above else 'DENTRO DO LIMITE'
        self.get_logger().info(f'[alerta] Pressao: {label}')

        audio_file = self._audio_above if is_above else self._audio_below

        if audio_file and os.path.isfile(audio_file):
            # Play configured MP3 with mpg123
            subprocess.Popen(
                ['mpg123', '-q', audio_file],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Fallback: text-to-speech via espeak-ng
            text = ('pressao acima do limite' if is_above
                    else 'pressao dentro do limite')
            subprocess.Popen(
                ['espeak-ng', '-v', 'pt', text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args=None):
    rclpy.init(args=args)
    node = AudioAlertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
