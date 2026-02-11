"""Unitree G1 scene configuration with box.xml asset."""

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg


def create_scene_cfg() -> SceneCfg:
    """Create a scene configuration with box.xml asset and unitree_g1 robot.
    
    Returns:
        SceneCfg instance with box and robot
    """
    # Path to the box.xml asset
    box_xml_path = MJLAB_SRC_PATH / "asset_zoo" / "box.xml"
    
    def add_box_asset(spec: mujoco.MjSpec) -> None:
        """Add the box.xml asset to the scene using spec_fn.
        
        This is called AFTER sensors are added, so we need to be careful
        not to cause conflicts. The box.xml itself doesn't have sensors,
        so this should be safe.
        """
        # Load the box XML as a separate spec
        box_spec = mujoco.MjSpec.from_file(str(box_xml_path))
        
        # Clear any sensors from box_spec to avoid conflicts
        # (box.xml shouldn't have sensors, but just to be safe)
        for sensor in list(box_spec.sensors):
            box_spec.delete(sensor)
        
        # Attach the box spec to the scene's worldbody
        # We'll attach it at a specific position (e.g., offset from robot)
        frame = spec.worldbody.add_frame()
        # Position the box next to the robot (e.g., 2 meters to the right)
        frame.pos = [2.0, 0.0, 0.0]
        spec.attach(box_spec, prefix="box/", frame=frame)
    
    # Configure contact sensors
    self_collision_sensor = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
    # Add feet_ground_contact sensor (required for reward terms)
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
    
    # Create scene configuration
    scene_cfg = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": get_g1_robot_cfg()},
        sensors=(self_collision_sensor, feet_ground_cfg),
        # Use spec_fn to add the box asset
        spec_fn=add_box_asset,
    )
    
    return scene_cfg


# Create the scene configuration
SCENE_CFG = create_scene_cfg()

