"""View the scene configuration with a viewer.

This script creates a minimal environment from the scene config and launches
a viewer to visualize it.
"""

import os
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viewer_config import ViewerConfig

from .scene_cfg import SCENE_CFG


def create_minimal_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create a minimal environment configuration for viewing the scene."""
    # Create a minimal action config (zero actions)
    action_cfg = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),  # Match all actuators
        scale=1.0,
        use_default_offset=True,
    )
    
    # Create environment config with the scene
    env_cfg = ManagerBasedRlEnvCfg(
        decimation=1,  # 1 physics step per environment step
        scene=SCENE_CFG,
        actions={"joint_pos": action_cfg},
        # Configure viewer to track the robot
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",  # Track the torso
            distance=3.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        episode_length_s=60.0,  # 60 second episodes
    )
    
    return env_cfg


def main():
    """Main function to create environment and launch viewer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View scene with box and unitree_g1")
    parser.add_argument(
        "--viewer",
        choices=["native", "viser", "auto"],
        default="auto",
        help="Viewer to use (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )
    
    args = parser.parse_args()
    
    # Create environment configuration
    print("Creating environment configuration...")
    env_cfg = create_minimal_env_cfg()
    
    # Create environment
    print("Creating environment...")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    
    # Wrap environment to provide get_observations() method for viewer
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    
    # Create a zero policy (no actions)
    class ZeroPolicy:
        def __call__(self, obs) -> torch.Tensor:
            action_shape = env.unwrapped.action_space.shape
            return torch.zeros(action_shape, device=env.unwrapped.device)
    
    policy = ZeroPolicy()
    
    # Determine which viewer to use
    if args.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        viewer_type = "native" if has_display else "viser"
    else:
        viewer_type = args.viewer
    
    # Launch the appropriate viewer
    print(f"Launching {viewer_type} viewer...")
    if viewer_type == "native":
        NativeMujocoViewer(env, policy).run()
    elif viewer_type == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise ValueError(f"Unknown viewer type: {viewer_type}")
    
    env.close()


if __name__ == "__main__":
    main()

