# 自定义 MDP 组件：奖励函数、课程学习与指令扩展

本文介绍如何在 go2_demo 项目中扩展 IsaacLab 的 MDP 框架，包括编写自定义奖励函数、实现课程学习、以及扩展速度指令配置。文中所有代码均来自项目实际代码，可直接参考。

---

## 目录

1. [整体思路](#1-整体思路)
2. [自定义奖励函数](#2-自定义奖励函数)
   - [2.1 编写奖励函数](#21-编写奖励函数)
   - [2.2 注册到环境配置](#22-注册到环境配置)
   - [2.3 自定义奖励计算逻辑（AER）](#23-自定义奖励计算逻辑aer)
3. [扩展速度指令](#3-扩展速度指令)
4. [课程学习](#4-课程学习)
   - [4.1 编写课程函数](#41-编写课程函数)
   - [4.2 注册到环境配置](#42-注册到环境配置)
5. [mdp 包的组织方式](#5-mdp-包的组织方式)
6. [扩展检查清单](#6-扩展检查清单)

---

## 1. 整体思路

IsaacLab 的 Manager-Based 框架把环境逻辑拆分成若干独立模块，每个模块（奖励、观测、指令……）都通过「函数 + 配置类」的方式定义。扩展时只需两步：

1. **写函数**：在 `mdp/` 目录下实现具体逻辑，接收 `env` 对象，返回 `torch.Tensor`。
2. **注册配置**：在环境配置类（`go2_demo_velocity.py`）里用对应的 `*Term` 配置类引用这个函数，框架会自动调用。

```
mdp/
├── __init__.py      ← 导出所有自定义函数，让环境配置里能直接用 mdp.xxx
├── rewards.py       ← 自定义奖励函数
├── curriculums.py   ← 自定义课程学习函数
└── commands.py      ← 扩展的指令配置类
```

---

## 2. 自定义奖励函数

### 2.1 编写奖励函数

**文件**：`mdp/rewards.py`

奖励函数的签名是固定的：第一个参数必须是 `env: ManagerBasedRLEnv`，其余参数通过配置时的 `params` 字典传入，返回形状为 `(num_envs,)` 的 Tensor。

**示例：能量效率奖励 `energy_new_actual`**

这个奖励来自学术论文，核心思想是「鼓励机器人用尽量少的能量跑得尽量快」。奖励值越接近 1，说明步态越节能。

```python
def energy_new_actual(
    env,
    asset_cfg=SceneEntityCfg("robot"),
    sigma_lin=1000.0,   # 线速度的能耗归一化系数，越大对能耗越宽容
    sigma_ang=500.0,    # 角速度的能耗归一化系数
    clip_lin=0.2,       # 线速度分母的最小值，防止除以零
    clip_ang=0.2,       # 角速度分母的最小值
):
    asset = env.scene[asset_cfg.name]

    # 实际消耗的能量：力矩 × 关节速度的绝对值之和
    joint_vel = asset.data.joint_vel
    joint_torque = asset.data.applied_torque
    energy = torch.sum(torch.abs(joint_vel * joint_torque), dim=1)

    # 分母：机器人实际线速度和角速度（clamp 防止速度接近零时分母塌陷）
    base_lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    base_ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    denom = (
        sigma_lin * torch.clamp(torch.abs(base_lin_vel_x), min=clip_lin)
        + sigma_ang * torch.clamp(torch.abs(base_ang_vel_z), min=clip_ang)
    )

    # exp(-energy/denom)：能耗越低，奖励越接近 1
    return torch.exp(-energy / denom)
```

**能量奖励的直觉理解：**

```
奖励 = exp(-消耗能量 / 有效运动量)
```

机器人跑得快且省力时，`energy / denom` 趋近 0，奖励趋近 1。
如果机器人抖动严重（能耗大但移动慢），奖励趋近 0。
`sigma_lin` 和 `sigma_ang` 控制对能耗的容忍程度，值越大越宽松。

**示例：脚部滑移惩罚 `feet_slip`**

```python
def feet_slip(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # 判断各脚是否处于接触状态（接触力 > 1 N 视为着地）
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1).max(dim=1)[0] > 1.0
    )

    # 只有着地的脚才计算滑移（速度的平方和）
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(contacts * torch.sum(torch.square(feet_vel_xy), dim=-1), dim=1)
```

这个惩罚项解决的问题是：机器人脚踩在地上时不应该横向滑动，否则在真实机器人上会打滑。只对「当前着地的脚」施加惩罚，腾空中的脚不算。

**编写自定义奖励函数时的要点：**

- 返回值必须是形状 `(num_envs,)` 的 Tensor，正数表示奖励，负数表示惩罚（最终由外部的 `weight` 决定正负方向）。
- 用 `env.scene["robot"]` 获取机器人资产，用 `env.scene.sensors["contact_forces"]` 获取传感器。
- `asset_cfg` 和 `sensor_cfg` 通过 `SceneEntityCfg` 传入，支持用 `body_names` 过滤特定身体部位。
- 常用数据接口：

| 数据 | 接口 |
|------|------|
| 关节角度 | `asset.data.joint_pos` |
| 关节速度 | `asset.data.joint_vel` |
| 输出力矩 | `asset.data.applied_torque` |
| 机体线速度（机器人坐标系） | `asset.data.root_lin_vel_b` |
| 机体角速度（机器人坐标系） | `asset.data.root_ang_vel_b` |
| 各刚体的世界线速度 | `asset.data.body_lin_vel_w` |
| 接触力历史 | `contact_sensor.data.net_forces_w_history` |

### 2.2 注册到环境配置

**文件**：`go2_demo_velocity.py`，`RewardsCfg` 类

```python
@configclass
class RewardsCfg:
    energy_new_actual = RewTerm(
        func=mdp.energy_new_actual,   # 指向自定义函数
        weight=0.8,                   # 正值=奖励，负值=惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma_lin": 1000.0,
            "sigma_ang": 500.0,
            "clip_lin": 0.2,
            "clip_ang": 0.2,
        },
    )

    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
```

`params` 里的键名必须和函数签名的参数名完全一致。`SceneEntityCfg` 支持用正则表达式（`body_names=".*_foot"`）过滤只取脚部刚体的数据。

### 2.3 自定义奖励计算逻辑（AER）

**文件**：`aer_env.py`

IsaacLab 默认把所有奖励项直接加权求和。但这种做法有个问题：如果某个惩罚项权重过大，机器人可能学会「死站原地」来规避惩罚，而不是去完成速度追踪任务。

AER（Adaptive Efficiency Reward）的解决思路是把正向奖励和惩罚分开处理：

```
总奖励 = Σ(正向奖励) × exp(Σ(惩罚) / σ_aux)
```

这样惩罚以指数的方式缩放正向奖励，而不是直接相减。惩罚越大，正向奖励被压缩得越多，但正向奖励永远不会变成负数——机器人不会因为惩罚太大就不行动。

```python
class AERRewardManager(RewardManager):
    # 声明哪些奖励项属于「正向激励」
    positive_terms = {"track_lin_vel_xy", "track_ang_vel_z", "energy_new_actual"}
    sigma_aux = 0.02   # 控制惩罚压缩的强度，越小惩罚越敏感

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        positive_reward = torch.zeros_like(self._reward_buf)
        negative_reward = torch.zeros_like(self._reward_buf)

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                continue
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

            if name in self.positive_terms:
                positive_reward += value
            else:
                negative_reward += value

        # 正向奖励被惩罚项以指数方式缩放
        self._reward_buf[:] = positive_reward * torch.exp(negative_reward / self.sigma_aux)
        return self._reward_buf
```

让这个自定义 RewardManager 生效，需要创建一个继承 `ManagerBasedRLEnv` 的子类，并重载 `load_managers` 方法：

```python
class AERManagerBasedRLEnv(ManagerBasedRLEnv):
    def load_managers(self):
        self.command_manager = CommandManager(self.cfg.commands, self)
        super().load_managers()
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.reward_manager = AERRewardManager(self.cfg.rewards, self)  # 替换为自定义版本
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        self._configure_gym_env_spaces()
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
```

最后在 `__init__.py` 里把 `entry_point` 指向这个子类：

```python
gym.register(
    id="Go2-velocity-v0",
    entry_point=f"{__name__}.aer_env:AERManagerBasedRLEnv",  # ← 这里
    ...
)
```

> 注意 `load_managers` 中各 Manager 的初始化顺序是有讲究的：Command Manager 要最先初始化（因为观测里有速度指令），RewardManager 要在 TerminationManager 之后（奖励计算依赖终止状态）。`super().load_managers()` 会初始化观测、动作和事件 Manager，不要重复调用。

---

## 3. 扩展速度指令

**文件**：`mdp/commands.py`

IsaacLab 内置的 `UniformVelocityCommandCfg` 只有一个速度范围 `ranges`。课程学习需要两个范围：一个是当前训练阶段的范围，一个是课程学习允许的最大范围（上限）。所以这里继承内置类，加入 `limit_ranges` 字段：

```python
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
```

`MISSING` 表示这个字段没有默认值，必须在使用时显式指定，否则会报错。使用方式见 `go2_demo_velocity.py` 的 `CommandsCfg`：

```python
base_velocity = mdp.UniformLevelVelocityCommandCfg(
    ...
    ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(0.2, 1.2),   # 初始训练阶段的速度范围
        ...
    ),
    limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(0.2, 6.0),   # 课程学习允许达到的最高速度
        ...
    ),
)
```

---

## 4. 课程学习

### 4.1 编写课程函数

**文件**：`mdp/curriculums.py`

课程函数的签名也是固定的：接收 `env` 和 `env_ids`（当前批次需要处理的环境索引），返回一个标量 Tensor 作为当前课程等级的指标（用于日志记录）。

```python
def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    # 拿到速度指令的配置对象（包含 ranges 和 limit_ranges）
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # 读取速度追踪奖励在本 episode 内的平均值
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    # 只在每个 episode 结束时更新一次课程等级
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            # 达到目标奖励的 80%，扩展速度范围（向更快的方向拓展 0.1 m/s）
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    # 返回当前速度上限作为课程等级指标（在 TensorBoard 里可以看到）
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)
```

**课程逻辑说明：**

升级条件：当前 episode 内 `track_lin_vel_xy` 的平均奖励 > 该项 `weight` 的 80%。

每次升级步长：`lin_vel_x` 范围两端各扩大 0.1 m/s，即速度区间整体向更高速拓展，同时被 `limit_ranges` 裁剪，不会超过上限。

不降级：这里只实现了单向升级。如果想加降级逻辑（机器人表现退步时缩小速度范围），可以在 `reward < threshold_low` 时做反向的 `delta_command`。

**课程进展的直观图示：**

```
初始范围:  lin_vel_x = [0.2, 1.2]
第一次升级: lin_vel_x = [0.1, 1.3]  （触碰下限 0.2，被 clamp 为 0.2）→ [0.2, 1.3]
第二次升级: lin_vel_x = [0.2, 1.4]
...
最终上限:  lin_vel_x = [0.2, 6.0]   （limit_ranges 约束）
```

### 4.2 注册到环境配置

**文件**：`go2_demo_velocity.py`

```python
@configclass
class CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)

@configclass
class GO2RobotDemoEnv(ManagerBasedRLEnvCfg):
    ...
    curriculum: CurriculumCfg = CurriculumCfg()   # ← 挂载到主环境配置
```

课程管理器（`CurriculumManager`）会在每个 episode 末尾自动调用 `lin_vel_cmd_levels`，无需手动触发。

---

## 5. mdp 包的组织方式

**文件**：`mdp/__init__.py`

所有自定义函数都要在 `__init__.py` 里导出，环境配置文件才能用 `mdp.xxx` 的方式引用：

```python
from isaaclab.envs.mdp import *   # 先导入 IsaacLab 内置的所有 MDP 函数

from .curriculums import *        # 再导入本项目的自定义函数（会覆盖同名内置函数）
from .commands import *
from .rewards import *
```

导入顺序很重要：自定义模块在后面导入，如果函数名和内置函数重名，自定义版本会覆盖内置版本。通常不要重名，除非是有意替换内置行为。

在环境配置文件里的导入方式：

```python
# go2_demo_velocity.py
from go2_demo.tasks.manager_based.go2_demo import mdp

# 然后就可以用 mdp.energy_new_actual, mdp.lin_vel_cmd_levels 等
```

---

## 6. 扩展检查清单

新增一个自定义组件时，按以下顺序操作：

**自定义奖励函数：**
- [ ] 在 `mdp/rewards.py` 里写函数，返回 `(num_envs,)` 的 Tensor
- [ ] 确认 `mdp/__init__.py` 有 `from .rewards import *`
- [ ] 在 `go2_demo_velocity.py` 的 `RewardsCfg` 里添加 `RewTerm`
- [ ] 如果是正向奖励，在 `aer_env.py` 的 `positive_terms` 集合里加上函数名

**自定义奖励计算逻辑：**
- [ ] 在 `aer_env.py` 里继承 `RewardManager`，重写 `compute` 方法
- [ ] 在同文件里继承 `ManagerBasedRLEnv`，重写 `load_managers`，替换 `reward_manager`
- [ ] 在 `__init__.py` 的 `gym.register` 里把 `entry_point` 指向自定义环境类

**自定义指令配置：**
- [ ] 在 `mdp/commands.py` 里继承 `UniformVelocityCommandCfg`，扩展字段
- [ ] 确认 `mdp/__init__.py` 有 `from .commands import *`
- [ ] 在 `go2_demo_velocity.py` 的 `CommandsCfg` 里使用新的配置类

**自定义课程学习：**
- [ ] 在 `mdp/curriculums.py` 里写函数，签名为 `(env, env_ids, ...) -> Tensor`
- [ ] 确认 `mdp/__init__.py` 有 `from .curriculums import *`
- [ ] 在 `go2_demo_velocity.py` 里添加 `CurriculumCfg` 类和 `CurrTerm`
- [ ] 在 `GO2RobotDemoEnv` 里挂载 `curriculum: CurriculumCfg = CurriculumCfg()`
- [ ] 确认使用的是 `AERManagerBasedRLEnv`（它初始化了 `CurriculumManager`）

---

## 参考资料

- [IsaacLab MDP API 文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab)（搜索 `mdp` → `rewards` / `curriculums` / `commands`）
- [IsaacLab Manager-Based 环境教程](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs)
- [RSL-RL PPO 实现](https://github.com/leggedrobotics/rsl_rl)
