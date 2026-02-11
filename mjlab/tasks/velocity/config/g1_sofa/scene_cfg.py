"""Unitree G1 scene configuration with sofa asset and two robots."""

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg


def create_scene_cfg() -> SceneCfg:
    """Create a scene configuration with sofa asset and two unitree_g1 robots.
    
    Returns:
        SceneCfg instance with sofa and two robots
    """
    # Path to the sofa asset
    sofa_path = MJLAB_SRC_PATH / "asset_zoo" / "objects" / "sofa" / "sofa.usdc"
    
    # Create EntityCfg for the sofa
    def sofa_spec_fn() -> mujoco.MjSpec:
        """Load the sofa USD file as a MuJoCo spec."""
        return mujoco.MjSpec.from_file(str(sofa_path))
    
    sofa_cfg = EntityCfg(
        spec_fn=sofa_spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.0),  # Position the sofa 2 meters to the right
        ),
    )
    
    # Get G1 robot config for both robots
    # Use "robot" as primary (for compatibility with env_cfg) and "robot2" as secondary
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.init_state.pos = (-1.0, 0.0, 0.0)  # Position robot to the left
    
    robot2_cfg = get_g1_robot_cfg()
    robot2_cfg.init_state.pos = (1.0, 0.0, 0.0)  # Position robot2 to the right
    
    # Configure contact sensors for the primary "robot" entity
    # (The env_cfg expects "robot" as the entity name)
    self_collision_sensor = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
    # Add feet_ground_contact sensor (required for reward terms)
    # Configured for the primary "robot" entity
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    
    # Create scene configuration with two robots and sofa
    scene_cfg = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={
            "robot": robot_cfg,
            "robot2": robot2_cfg,
            "sofa": sofa_cfg,
        },
        sensors=(self_collision_sensor, feet_ground_cfg),
    )
    
    return scene_cfg


# Create the scene configuration
SCENE_CFG = create_scene_cfg()

