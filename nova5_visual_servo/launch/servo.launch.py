from launch import LaunchDescription
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([
        Node(package='nova5_visual_servo', executable='vision', output='screen'),
        Node(package='nova5_visual_servo', executable='servo', output='screen')])
