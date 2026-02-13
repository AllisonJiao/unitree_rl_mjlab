"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "prettytable",
    "tqdm",
    "tyro>=1.0.1",
    "torch>=2.7.0",
    "torchrunx>=0.3.4",
    "warp-lang>=1.11.0.dev20251211",
    # mujoco-warp: use >=3.5.0 or install from git main if you see flexedge_J AttributeError
    # (MuJoCo moved flexedge from MjData to MjModel; fix in mujoco_warp PR #1015)
    "mujoco-warp",
    "mujoco>=3.4.0",
    "trimesh>=4.8.3",
    "viser>=1.0.16",
    "moviepy",
    "tensordict",
    "rsl-rl-lib==3.1.0",
    "tensorboard>=2.20.0",
    "onnxscript>=0.5.4",
    "wandb>=0.22.3",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["mjlab"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
