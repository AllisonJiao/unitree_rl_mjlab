"""Sofa asset constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

SOFA_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "objects" / "sofa" / "xmls" / "sofa.xml"
)
assert SOFA_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, SOFA_XML.parent / "asset", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SOFA_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

DEFAULT_STATE = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0),
  # No joints on the sofa, so set joint_pos/joint_vel to None.
  joint_pos=None,
  joint_vel=None,
)

##
# Final config.
##


def get_sofa_cfg() -> EntityCfg:
  """Get a fresh sofa configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  The sofa is a fixed-base, non-articulated entity (static prop).
  """
  return EntityCfg(
    init_state=DEFAULT_STATE,
    spec_fn=get_spec,
    # No articulation — the sofa is a static object.
    articulation=None,
  )

