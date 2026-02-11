"""Environment configuration for viewing the scene with box and unitree_g1.

Supports both zero/random agents and trained checkpoints from G1 velocity tasks.
"""

import math

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
    
    # Replace the scene with our custom scene (box + robot)
    env_cfg.scene = SCENE_CFG
    
    # Update action config to match G1 velocity task
    joint_pos_action = env_cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE
    
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

