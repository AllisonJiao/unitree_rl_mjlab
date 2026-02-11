from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .env_cfg_scene import scene_view_env_cfg
from .rl_cfg import unitree_g1_ppo_runner_cfg
from .rl_cfg_scene import scene_view_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Register scene viewing task
register_mjlab_task(
  task_id="Mjlab-Scene-View-Box-G1",
  env_cfg=scene_view_env_cfg(play=False),
  play_env_cfg=scene_view_env_cfg(play=True),
  rl_cfg=scene_view_ppo_runner_cfg(),
  runner_cls=None,  # Use default OnPolicyRunner
)
