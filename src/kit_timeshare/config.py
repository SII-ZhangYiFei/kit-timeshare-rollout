from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommonConfig:
    num_envs: int = 2
    gpu_uuid: str | None = None
    output_dir: str | Path = "timeshare-run"
    seed: int = 0
    verify_frozen: bool = False
    max_episode_steps: int | None = None
    headless: bool = True
    strict_versions: bool = True
    cpu_threads: int = 8

    def __post_init__(self):
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.max_episode_steps is not None and (
            type(self.max_episode_steps) is not int or self.max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be positive or None")
        if type(self.cpu_threads) is not int or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")


@dataclass(frozen=True)
class BehaviorConfig(CommonConfig):
    config_path: str | Path | None = None
    config: dict | None = None
    upstream_root: str | Path | None = None
    scene_spacing: float = 40.0
    read_only_material_queries: bool = True
    descriptor_sets: int = 360000
    reserved_descriptors: int = 900000


@dataclass(frozen=True)
class RoboDojoConfig(CommonConfig):
    upstream_root: str | Path = "."
    task: str = "stack_bowls"
    benchmark: str = "RoboDojo"
    robot_config: str = "env_cfg/robot/dual_x5.yml"
    eval_config: str = "env_cfg/arx_x5.yml"
    sim_config: str = "env_cfg/sim/sim_config.yml"
    scene_config: str = "env_cfg/scene/default.yml"
    camera_config: str = "env_cfg/camera/camera_config.yml"
    layout_path: str | Path | None = None
    control_substeps: int = 10
    experience: str | Path | None = None
    urdf_importer_version: str | None = "2.4.30"

    def __post_init__(self):
        super().__post_init__()
        if type(self.control_substeps) is not int or self.control_substeps < 1:
            raise ValueError("control_substeps must be positive")
        if not self.task.isidentifier() or not self.benchmark.isidentifier():
            raise ValueError("task and benchmark must be Python identifiers")
