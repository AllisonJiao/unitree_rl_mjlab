"""Environment configuration for viewing the scene with sofa and unitree_g1.

Supports both zero/random agents and trained checkpoints from G1 velocity tasks.
"""

import math

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.viewer.viewer_config import ViewerConfig

from .scene_cfg import SCENE_CFG


def scene_view_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create environment configuration for viewing the scene.
    
    This config supports:
    - Zero/random agents (for simple viewing)
    - Trained checkpoints from G1 velocity tasks (includes observations and commands)
    
    Args:
        play: If True, configure for play mode (e.g., better viewer settings)
    
    Returns:
        ManagerBasedRlEnvCfg with the scene setup
    """
    # Start with base velocity env config to get observations and commands
    env_cfg = make_velocity_env_cfg()
    
    # Replace the scene with our custom scene (sofa + robot)
    env_cfg.scene = SCENE_CFG
    
    # Remove curriculum since we're using a simple plane terrain (no terrain generator)
    # The curriculum requires a terrain generator which we don't have
    env_cfg.curriculum = {}
    
    # Update action config to match G1 velocity task
    joint_pos_action = env_cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE
    
    # Note: feet_ground_contact sensor is already in SCENE_CFG, so we don't need to add it here
    site_names = ("left_foot", "right_foot")
    
    # Configure reward parameters that require robot-specific settings
    # These are needed for the reward functions to work properly
    
    # Configure pose reward std values (required for variable_posture reward)
    env_cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    env_cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.15,
        r".*ankle_roll.*": 0.1,
        # Waist.
        r".*waist_yaw.*": 0.15,
        r".*waist_roll.*": 0.1,
        r".*waist_pitch.*": 0.1,
        # Arms.
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }
    env_cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.25,
        r".*hip_yaw.*": 0.25,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        # Waist.
        r".*waist_yaw.*": 0.25,
        r".*waist_roll.*": 0.1,
        r".*waist_pitch.*": 0.1,
        # Arms.
        r".*shoulder_pitch.*": 0.25,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }
    
    # Configure other reward parameters
    env_cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
    env_cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
    env_cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
    
    # Add self_collisions reward (uses the self_collision sensor from scene_cfg)
    env_cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    
    # Configure event parameters
    env_cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    
    # Configure viewer to track the robot
    env_cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",  # Track the torso
        distance=3.0,
        elevation=-20.0,
        azimuth=45.0,
    )
    
    # Update episode length
    env_cfg.episode_length_s = 60.0
    
    # For play mode, you might want to adjust command settings
    if play:
        # Make commands less aggressive for viewing
        twist_cmd = env_cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.3, 0.8)  # Slower forward speed
        twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
        twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)  # Slower rotation
    
    return env_cfg

