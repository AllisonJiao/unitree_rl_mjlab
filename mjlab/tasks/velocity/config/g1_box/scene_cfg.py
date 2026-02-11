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
        """Add the box.xml asset to the scene using spec_fn."""
        # Load the box XML as a separate spec
        box_spec = mujoco.MjSpec.from_file(str(box_xml_path))
        
        # Attach the box spec to the scene's worldbody
        # We'll attach it at a specific position (e.g., offset from robot)
        frame = spec.worldbody.add_frame()
        # Position the box next to the robot (e.g., 2 meters to the right)
        frame.pos = [2.0, 0.0, 0.0]
        spec.attach(box_spec, prefix="box/", frame=frame)
    
    # Configure contact sensor
    self_collision_sensor = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
    # Create scene configuration
    scene_cfg = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": get_g1_robot_cfg()},
        sensors=(self_collision_sensor,),
        # Use spec_fn to add the box asset
        spec_fn=add_box_asset,
    )
    
    return scene_cfg


# Create the scene configuration
SCENE_CFG = create_scene_cfg()

