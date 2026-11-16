# Kit 时分 rollout

在同一个 Kit 进程中，让多个 **CPU PhysX + GPU RTX 渲染**环境轮流推进。
`pool.step(env_id, action)` 接收你的策略动作，返回观测、原生奖励、终止标志和信息；其他环境保留各自的物理状态、任务状态和时钟。

这是 BEHAVIOR-1K/OmniGibson 与 RoboDojo 的实验性适配层。需要先安装上游 SDK 和数据，不是重新实现这两个 benchmark，也不执行 GPU checkpoint 或显存 offload。

```bash
# 在已经可以运行上游 benchmark 的 Python 环境中安装
python -m pip install git+https://github.com/SII-ZhangYiFei/kit-timeshare-rollout.git
kit-timeshare doctor
```

```python
from kit_timeshare import BehaviorConfig, make

with make(BehaviorConfig(num_envs=2, config_path="my_behavior.yaml",
                         output_dir="new-run", verify_frozen=True)) as pool:
    observations = [pool.observe(e) for e in range(pool.num_envs)]
    for turn in range(100):
        e = turn % pool.num_envs
        result = pool.step(e, policy(observations[e]))
        observations[e] = result.observation
        if result.terminated or result.truncated:
            observations[e], info = pool.reset(e)
```

RoboDojo 改用 `RoboDojoConfig(upstream_root="/path/to/RoboDojo")`，动作采用上游 `MetaControl` 或其字典。完整策略入口见 [examples/rollout.py](examples/rollout.py)。

- SDK 版本、上游提交和已测范围见[兼容矩阵](src/kit_timeshare/compatibility.json)与[验收记录](docs/validation.md)。
- 多卡机器必须隔离目标 GPU 和对应 DRM 渲染节点，单设 `CUDA_VISIBLE_DEVICES` 无法约束 Vulkan；`kit-timeshare docker-command` 可生成隔离启动命令。
- 不自动重置、不产生默认策略动作、不人为抬高物体或施加测试初速度。
- reset 仅支持物体集合固定的场景；RoboDojo 恢复已加载布局，不重新随机采样布局。
- 单进程只创建一个 pool；在创建它的线程中调用 Kit。`verify_frozen=True` 会增加检查开销，发现其他环境变化就停止。
- 旧实验的 42/31 环境数量不代表本版本在任意任务上都能容纳这些数量；新接口的已测范围以验收记录为准。

详见 [English README](README.md) 与 [API 限制](docs/api.md)。
