"""Environment configuration for viewing the scene with box and unitree_g1."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.viewer.viewer_config import ViewerConfig

from .scene_cfg import SCENE_CFG


def scene_view_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create environment configuration for viewing the scene.
    
    Args:
        play: If True, configure for play mode (e.g., better viewer settings)
    
    Returns:
        ManagerBasedRlEnvCfg with the scene setup
    """
    # Create a minimal action config (zero actions)
    action_cfg = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),  # Match all actuators
        scale=1.0,
        use_default_offset=True,
    )
    
    # Configure viewer to track the robot
    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",  # Track the torso
        distance=3.0,
        elevation=-20.0,
        azimuth=45.0,
    )
    
    # Create environment config with the scene
    env_cfg = ManagerBasedRlEnvCfg(
        decimation=1,  # 1 physics step per environment step
        scene=SCENE_CFG,
        actions={"joint_pos": action_cfg},
        viewer=viewer_cfg,
        episode_length_s=60.0,  # 60 second episodes
    )
    
    return env_cfg

