"""Unitree G1 scene configuration with sofa asset and two robots."""

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg


def create_scene_cfg() -> SceneCfg:
    """Create a scene configuration with sofa asset and two unitree_g1 robots.
    
    Returns:
        SceneCfg instance with sofa and two robots
    """
    # Path to the sofa XML file
    sofa_xml_path = MJLAB_SRC_PATH / "asset_zoo" / "objects" / "sofa" / "sofa.xml"
    
    def add_sofa_asset(spec: mujoco.MjSpec) -> None:
        """Add the sofa XML asset to the scene using spec_fn.
        
        This is called AFTER sensors are added, so we need to be careful
        not to cause conflicts.
        """
        # Load the sofa XML as a separate spec
        sofa_spec = mujoco.MjSpec.from_file(str(sofa_xml_path))
        
        # Clear any sensors from sofa_spec to avoid conflicts
        for sensor in list(sofa_spec.sensors):
            sofa_spec.delete(sensor)
        
        # Attach the sofa spec to the scene's worldbody
        frame = spec.worldbody.add_frame()
        # Position the sofa 2 meters to the right
        frame.pos = [2.0, 0.0, 0.0]
        spec.attach(sofa_spec, prefix="sofa/", frame=frame)
    
    # Get G1 robot config for both robots
    # Use "robot" as primary (for compatibility with env_cfg) and "robot2" as secondary
    # Create a fresh copy to avoid any mutation issues
    from copy import deepcopy
    
    robot_cfg = get_g1_robot_cfg()
    # Create a new InitialStateCfg with updated position to avoid mutating the original
    robot_cfg.init_state = deepcopy(robot_cfg.init_state)
    robot_cfg.init_state.pos = (-1.0, 0.0, 0.0)  # Position robot to the left
    
    robot2_cfg = get_g1_robot_cfg()
    # Create a new InitialStateCfg with updated position
    robot2_cfg.init_state = deepcopy(robot2_cfg.init_state)
    robot2_cfg.init_state.pos = (1.0, 0.0, 0.0)  # Position robot2 to the right
    # Remove actuators from robot2 so it doesn't interfere with observations/actions
    # This makes robot2 passive (no control, just physics)
    if robot2_cfg.articulation is not None:
        robot2_cfg.articulation = deepcopy(robot2_cfg.articulation)
        robot2_cfg.articulation.actuators = tuple()
    
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
    # Note: Temporarily comment out robot2 to debug CUDA error
    # Uncomment robot2 once the issue is resolved
    scene_cfg = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={
            "robot": robot_cfg,
            # "robot2": robot2_cfg,  # Temporarily disabled to debug CUDA error
        },
        sensors=(self_collision_sensor, feet_ground_cfg),
        # Use spec_fn to add the sofa asset
        spec_fn=add_sofa_asset,
    )
    
    return scene_cfg


# Create the scene configuration
SCENE_CFG = create_scene_cfg()

