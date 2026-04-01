"""Launch file for the X3Plus MoveIt pick-and-place stack.

Starts:
  1. robot_state_publisher   (URDF with topic_based_ros2_control)
  2. controller_manager      (ros2_control with TopicBasedSystem)
  3. joint_trajectory_controller + x3plus_gripper_controller
  4. move_group              (MoveIt2 planning + execution)
  5. pick_place_node         (BT + analytical IK orchestration)
"""

import os
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("x3plus_pick_place")
    desc_share = get_package_share_directory("x3plus_description")

    urdf_path = os.path.join(desc_share, "urdf", "x3plus.urdf")
    rc_xacro_path = os.path.join(desc_share, "urdf", "x3plus.ros2_control.xacro")
    srdf_path = os.path.join(desc_share, "config", "x3plus.srdf")
    kinematics_path = os.path.join(desc_share, "config", "kinematics.yaml")
    joint_limits_path = os.path.join(desc_share, "config", "joint_limits.yaml")
    ompl_path = os.path.join(desc_share, "config", "ompl_planning.yaml")
    controllers_yaml_path = os.path.join(pkg_share, "config", "controllers.yaml")

    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()
    rc_tree = ET.parse(rc_xacro_path)
    rc_root = rc_tree.getroot()
    for rc_elem in rc_root.findall("ros2_control"):
        urdf_root.append(rc_elem)

    robot_description = {
        "robot_description": ET.tostring(urdf_root, encoding="unicode")
    }
    robot_description_semantic = {
        "robot_description_semantic": _read_text(srdf_path)
    }
    robot_description_kinematics = {
        "robot_description_kinematics": _load_yaml(kinematics_path)
    }
    robot_description_planning = {
        "robot_description_planning": _load_yaml(joint_limits_path)
    }

    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": _load_yaml(ompl_path),
    }
    trajectory_execution = {
        "moveit_simple_controller_manager": {
            "controller_names": ["joint_trajectory_controller"],
            "joint_trajectory_controller": {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": [f"arm_joint{i}" for i in range(1, 6)],
            },
        },
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 2.0,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.05,
    }
    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

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
            spawn_arm_controller,
            spawn_gripper_controller,
        ],
    )
    delayed_move_group = TimerAction(period=12.0, actions=[move_group_node])
    delayed_pick_place = TimerAction(period=20.0, actions=[pick_place_node])
    shutdown_on_pick_place_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=pick_place_node,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return LaunchDescription([
        static_tf,
        robot_state_publisher,
        controller_manager,
        delayed_spawners,
        delayed_move_group,
        delayed_pick_place,
        shutdown_on_pick_place_exit,
    ])
