"""Unitree G1 scene configuration.

To view this scene, run:
    python -m mjlab.tasks.velocity.config.g1.view_scene

Or from the config directory:
    python view_scene.py
"""

from mjlab.scene import SceneCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg

# Configure contact sensor
self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)

# Create scene
SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    entities={"robot": get_g1_robot_cfg()},
    sensors=(self_collision_sensor,),
)