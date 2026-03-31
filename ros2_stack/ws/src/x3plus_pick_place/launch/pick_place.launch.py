"""Launch file for the X3Plus pick-and-place (direct-control mode).

Only starts the BT orchestrator node. The MuJoCo bridge must be running
separately — it provides /joint_states, /cube_pose, /target_place_pose
and accepts /joint_command.

No MoveIt, controller_manager, or ros2_control stack needed.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pick_place_node = Node(
        package="x3plus_pick_place",
        executable="pick_place_node",
        output="screen",
    )

    return LaunchDescription([pick_place_node])
