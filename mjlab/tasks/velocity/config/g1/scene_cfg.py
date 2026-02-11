"""Unitree G1 scene configuration."""
from dataclasses import replace

from mjlab.scene import SceneCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.utils.spec_config import ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg

# Configure contact sensor
self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    subtree1="pelvis",
    subtree2="pelvis",
    data=("found",),
    reduce="netforce",
    num=10,  # report up to 10 contacts
)

# Add sensor to robot config
g1_cfg = replace(get_g1_robot_cfg(), sensors=(self_collision_sensor,))

# Create scene
SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    entities={"robot": g1_cfg},
)