"""Launch file for the X3Plus two-block pick-and-place.

Only starts the BT orchestrator node. The MuJoCo bridge must be running
separately — it provides /joint_states, /yellow_block_pose, /red_block_pose
and accepts /joint_command.

BT configuration is loaded from config/pick_place_tree.yaml (installed into
the package share directory).
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
