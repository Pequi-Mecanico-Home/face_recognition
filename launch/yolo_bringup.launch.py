import os

from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node



def generate_launch_description():

    realsense = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('realsense2_camera'),'launch','rs_launch.py'
                )])
    )


    yolo = Node(package='yolo_pose', executable='inference',
                        output='screen')

    return LaunchDescription([
        realsense,
        yolo,
    ])