"""Launch file for the X3Plus pick-and-place ROS2 stack.

Starts:
  1. robot_state_publisher   (URDF with topic_based_ros2_control)
  2. controller_manager      (ros2_control with TopicBasedSystem)
  3. joint_trajectory_controller + x3plus_gripper_controller (spawned)
  4. move_group              (MoveIt2 planning, position-only IK)
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
    pkg_share = get_package_share_directory("x3plus_pick_place")
    desc_share = get_package_share_directory("x3plus_description")

    # ── Robot Description (URDF + ros2_control hardware) ──────────────────
    urdf_path = os.path.join(desc_share, "urdf", "x3plus.urdf")
    rc_xacro_path = os.path.join(desc_share, "urdf", "x3plus.ros2_control.xacro")

    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()

    # Remove mimic from gripper linkage joints so ros2_control doesn't choke
    for jname in ("rlink_joint2", "rlink_joint3", "llink_joint1", "llink_joint2", "llink_joint3"):
        for joint in urdf_root.findall(f".//joint[@name='{jname}']"):
            mimic_elem = joint.find("mimic")
            if mimic_elem is not None:
                joint.remove(mimic_elem)

    # Inject topic_based_ros2_control hardware
    rc_tree = ET.parse(rc_xacro_path)
    rc_root = rc_tree.getroot()
    for rc_elem in rc_root.findall("ros2_control"):
        urdf_root.append(rc_elem)

    robot_description_content = ET.tostring(urdf_root, encoding="unicode")
    robot_description = {"robot_description": robot_description_content}

    # ── SRDF ──────────────────────────────────────────────────────────────
    srdf_path = os.path.join(desc_share, "config", "x3plus.srdf")
    with open(srdf_path, "r") as f:
        srdf_content = f.read()
    robot_description_semantic = {"robot_description_semantic": srdf_content}

    # ── Kinematics ────────────────────────────────────────────────────────
    kinematics_path = os.path.join(desc_share, "config", "kinematics.yaml")
    with open(kinematics_path, "r") as f:
        kinematics_yaml = yaml.safe_load(f)
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    # ── Joint limits (inferred from URDF) ─────────────────────────────────
    joint_limits_yaml = {
        "joint_limits": {
            "arm_joint1": {"has_velocity_limits": True, "max_velocity": 1.0,
                           "has_acceleration_limits": True, "max_acceleration": 2.0},
            "arm_joint2": {"has_velocity_limits": True, "max_velocity": 1.0,
                           "has_acceleration_limits": True, "max_acceleration": 2.0},
            "arm_joint3": {"has_velocity_limits": True, "max_velocity": 1.0,
                           "has_acceleration_limits": True, "max_acceleration": 2.0},
            "arm_joint4": {"has_velocity_limits": True, "max_velocity": 1.0,
                           "has_acceleration_limits": True, "max_acceleration": 2.0},
            "arm_joint5": {"has_velocity_limits": True, "max_velocity": 1.0,
                           "has_acceleration_limits": True, "max_acceleration": 2.0},
        }
    }
    robot_description_planning = {"robot_description_planning": joint_limits_yaml}

    # ── Controllers config ────────────────────────────────────────────────
    controllers_yaml_path = os.path.join(pkg_share, "config", "controllers.yaml")

    # ── MoveIt2 planning pipeline ─────────────────────────────────────────
    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {},
    }

    trajectory_execution = {
        "moveit_simple_controller_manager": {
            "controller_names": ["joint_trajectory_controller", "x3plus_gripper_controller"],
            "joint_trajectory_controller": {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": [f"arm_joint{i}" for i in range(1, 6)],
            },
            "x3plus_gripper_controller": {
                "type": "GripperCommand",
                "action_ns": "gripper_cmd",
                "default": True,
                "joints": ["grip_joint"],
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

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_footprint"],
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
        parameters=[robot_description, controllers_yaml_path],
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

    spawn_gripper_controller = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "x3plus_gripper_controller",
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
        package="x3plus_pick_place",
        executable="pick_place_node",
        output="screen",
    )

    delayed_spawners = TimerAction(
        period=5.0,
        actions=[
            spawn_joint_state_broadcaster,
            spawn_arm_controller,
            spawn_gripper_controller,
        ],
    )

    delayed_move_group = TimerAction(
        period=12.0,
        actions=[move_group_node],
    )

    delayed_pick_place = TimerAction(
        period=20.0,
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
