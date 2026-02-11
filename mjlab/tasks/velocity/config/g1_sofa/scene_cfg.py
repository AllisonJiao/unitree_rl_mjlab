"""Unitree G1 scene configuration with sofa asset and two robots."""

from copy import deepcopy

from mjlab.asset_zoo.objects.sofa.sofa_constants import get_sofa_cfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg


def create_scene_cfg() -> SceneCfg:
    """Create a scene configuration with sofa asset and two unitree_g1 robots.
    
    Returns:
        SceneCfg instance with sofa and two robots
    """
    # --- Sofa entity ---
    sofa_cfg = get_sofa_cfg()
    sofa_cfg.init_state.pos = (2.0, 0.0, 0.0)  # Position sofa to the right
    
    # --- Primary robot (controlled) ---
    robot_cfg = get_g1_robot_cfg()
    # Preserve the z-coordinate from HOME_KEYFRAME (0.78) to keep robot above ground
    robot_cfg.init_state.pos = (-1.0, 0.0, 0.78)
    
    # --- Secondary robot (passive) ---
    robot2_cfg = get_g1_robot_cfg()
    robot2_cfg.init_state.pos = (1.0, 0.0, 0.78)
    # Deep copy articulation before clearing actuators to avoid mutating the shared G1_ARTICULATION
    if robot2_cfg.articulation is not None:
        robot2_cfg.articulation = deepcopy(robot2_cfg.articulation)
        robot2_cfg.articulation.actuators = tuple()
    
    # --- Contact sensors (primary "robot" entity only) ---
    self_collision_sensor = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
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
    
    # --- Scene ---
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

