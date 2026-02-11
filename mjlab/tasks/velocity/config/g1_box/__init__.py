from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import scene_view_env_cfg
from .rl_cfg import scene_view_ppo_runner_cfg

# Register scene viewing task
register_mjlab_task(
  task_id="Mjlab-Scene-View-Box-G1",
  env_cfg=scene_view_env_cfg(play=False),
  play_env_cfg=scene_view_env_cfg(play=True),
  rl_cfg=scene_view_ppo_runner_cfg(),
  runner_cls=None,  # Use default OnPolicyRunner
)