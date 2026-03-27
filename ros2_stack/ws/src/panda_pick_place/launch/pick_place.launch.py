"""Launch file for the Franka pick-and-place ROS2 stack.

Starts:
  1. robot_state_publisher   (URDF with topic_based_ros2_control)
  2. controller_manager      (ros2_control with TopicBasedSystem)
  3. joint_trajectory_controller + panda_hand_controller (spawned)
  4. move_group              (MoveIt2 planning)
  5. pick_place_node         (BT orchestrator)
"""

import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

import yaml


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("panda_pick_place")
    panda_description_pkg = get_package_share_directory("moveit_resources_panda_description")
    panda_moveit_pkg = get_package_share_directory("moveit_resources_panda_moveit_config")

    # ── Robot Description (URDF + ros2_control hardware) ──────────────────
    urdf_path = os.path.join(panda_description_pkg, "urdf", "panda.urdf")
    rc_xacro_path = os.path.join(pkg_share, "config", "panda.ros2_control.xacro")

    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()

    # Remove mimic attribute from panda_finger_joint2 to satisfy ros2_control
    for joint in urdf_root.findall(".//joint[@name='panda_finger_joint2']"):
        mimic_elem = joint.find("mimic")
        if mimic_elem is not None:
            joint.remove(mimic_elem)

    rc_tree = ET.parse(rc_xacro_path)
    rc_root = rc_tree.getroot()
    for rc_elem in rc_root.findall("ros2_control"):
        urdf_root.append(rc_elem)

    robot_description_content = ET.tostring(urdf_root, encoding="unicode")
    robot_description = {"robot_description": robot_description_content}

    # ── SRDF (with extra ACM entries for panda near-link collisions) ─────
    srdf_path = os.path.join(panda_moveit_pkg, "config", "panda.srdf")
    with open(srdf_path, "r") as f:
        srdf_content = f.read()

    extra_acm = (
        '  <disable_collisions link1="panda_hand" link2="panda_link5" reason="Never"/>\n'
        '  <disable_collisions link1="panda_link5" link2="panda_link7" reason="Never"/>\n'
    )
    srdf_content = srdf_content.replace("</robot>", extra_acm + "</robot>")
    robot_description_semantic = {"robot_description_semantic": srdf_content}

    # ── Kinematics ────────────────────────────────────────────────────────
    kinematics_yaml_path = os.path.join(panda_moveit_pkg, "config", "kinematics.yaml")
    with open(kinematics_yaml_path, "r") as f:
        kinematics_yaml = yaml.safe_load(f)
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    # ── Joint limits ────────────────────────────────────────────────────
    joint_limits_yaml_path = os.path.join(panda_moveit_pkg, "config", "joint_limits.yaml")
    with open(joint_limits_yaml_path, "r") as f:
        joint_limits_yaml = yaml.safe_load(f)
    robot_description_planning = {"robot_description_planning": joint_limits_yaml}

    # ── Controllers config ────────────────────────────────────────────────
    controllers_yaml_path = os.path.join(pkg_share, "config", "controllers.yaml")
    with open(controllers_yaml_path, "r") as f:
        controllers_yaml = yaml.safe_load(f)

    # ── MoveIt2 planning pipeline ─────────────────────────────────────────
    ompl_yaml_path = os.path.join(panda_moveit_pkg, "config", "ompl_planning.yaml")
    ompl_yaml = {}
    if os.path.exists(ompl_yaml_path):
        with open(ompl_yaml_path, "r") as f:
            ompl_yaml = yaml.safe_load(f) or {}

    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_yaml,
    }

    trajectory_execution = {
        "moveit_simple_controller_manager": {
            "controller_names": ["joint_trajectory_controller", "panda_hand_controller"],
            "joint_trajectory_controller": {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": [f"panda_joint{i}" for i in range(1, 8)],
            },
            "panda_hand_controller": {
                "type": "GripperCommand",
                "action_ns": "gripper_cmd",
                "default": True,
                "joints": ["panda_finger_joint1"],
            },
        },
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 2.0,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.0,
    }

    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # ── Nodes ─────────────────────────────────────────────────────────────

    # Static TF: world -> panda_link0 (required by MoveIt2 virtual_joint)
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, controllers_yaml],
        output="screen",
    )

    spawn_joint_state_broadcaster = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "joint_state_broadcaster",
        ],
        output="screen",
    )

    spawn_arm_controller = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "joint_trajectory_controller",
        ],
        output="screen",
    )

    spawn_hand_controller = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "panda_hand_controller",
        ],
        output="screen",
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            planning_pipelines,
            trajectory_execution,
            planning_scene_monitor,
        ],
    )

    pick_place_node = Node(
        package="panda_pick_place",
        executable="pick_place_node",
        output="screen",
    )

    delayed_spawners = TimerAction(
        period=3.0,
        actions=[
            spawn_joint_state_broadcaster,
            spawn_arm_controller,
            spawn_hand_controller,
        ],
    )

    delayed_move_group = TimerAction(
        period=8.0,
        actions=[move_group_node],
    )

    delayed_pick_place = TimerAction(
        period=15.0,
        actions=[pick_place_node],
    )

    return LaunchDescription([
        static_tf,
        robot_state_publisher,
        controller_manager,
        delayed_spawners,
        delayed_move_group,
        delayed_pick_place,
    ])
