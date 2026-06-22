# Go2 机器狗速度跟随训练指南

基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab) + 宇树 Go2 机器人的 PPO 强化学习训练项目。任务目标是让机器狗跟踪给定的线速度和角速度指令在平坦地面上行走。

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目结构](#2-项目结构)
3. [代码实现](#3-代码实现)
   - [3.1 机器人资产配置](#31-机器人资产配置)
   - [3.2 场景配置](#32-场景配置)
   - [3.3 观测空间](#33-观测空间)
   - [3.4 动作空间](#34-动作空间)
   - [3.5 速度指令](#35-速度指令)
   - [3.6 奖励函数](#36-奖励函数)
   - [3.7 终止条件](#37-终止条件)
   - [3.8 域随机化](#38-域随机化)
   - [3.9 课程学习](#39-课程学习)
   - [3.10 注册 Gym 环境](#310-注册-gym-环境)
   - [3.11 PPO 训练超参数](#311-ppo-训练超参数)
4. [训练与测试](#4-训练与测试)
5. [调试技巧](#5-调试技巧)

---

## 1. 环境准备

### 安装 IsaacSim 和 IsaacLab

参考 [IsaacLab 官方仓库](https://github.com/isaac-sim/IsaacLab) 完成安装。IsaacSim 需要 NVIDIA GPU，建议显存 16 GB 以上。

### 创建外部项目

在 IsaacLab 根目录执行以下命令，它会基于模板生成一个独立的外部项目骨架：

```bash
./isaaclab.sh -n
```

按提示命名项目（本项目命名为 `go2_demo`），完成后进入项目根目录安装：

```bash
cd go2_demo
python -m pip install -e source/go2_demo
```

`-e` 参数表示可编辑安装，修改代码后无需重新安装即可生效。

### 准备机器人模型

从 [HuggingFace](https://huggingface.co/datasets/unitreerobotics/unitree_model) 下载宇树开源机器人模型文件：

- **新版 IsaacLab**：可以直接使用 URDF 文件，通过 `UrdfFileCfg` 加载。
- **旧版 IsaacLab**：需要先用 Isaac Sim 自带工具将 URDF 转换为 USD 格式，再通过 `UsdFileCfg` 加载。

**在 Isaac Sim 中查看 USD 文件**：点击菜单栏 `File > Open` 直接打开，或从 Content Tab 把文件拖入 Viewport。

将下载好的模型目录拷贝到 `source/go2_demo/go2_demo/assets/robot/unitree_model`。

---

## 2. 项目结构

```
go2_demo/
├── scripts/                          # 训练、测试相关脚本
│   ├── zero_agent.py                 # 发送零动作，验证环境是否能正常启动
│   ├── random_agent.py               # 发送随机动作，检查机器人状态是否异常
│   └── rsl_rl/
│       ├── train.py                  # 训练入口
│       └── play.py                   # 回放 / 导出策略
├── source/go2_demo/go2_demo/
│   ├── assets/robot/
│   │   ├── unitree.py                # 机器人资产配置（ArticulationCfg）
│   │   └── unitree_model/Go2/        # USD 模型文件
│   └── tasks/manager_based/go2_demo/
│       ├── __init__.py               # 注册 Gym 环境
│       ├── go2_demo_velocity.py      # 主环境配置（场景、观测、奖励等）
│       ├── aer_env.py                # 自定义 ManagerBasedRLEnv（AER 奖励机制）
│       ├── mdp/                      # 自定义 MDP 函数（观测、奖励、指令等）
│       └── agents/
│           └── rsl_rl_ppo_cfg.py     # PPO 训练超参数
└── logs/                             # 训练日志和保存的模型
```

---

## 3. 代码实现

整个环境配置基于 IsaacLab 的 **Manager-Based** 框架。这套框架把环境拆分成若干独立的「Manager」——每个 Manager 负责一件事（观测、动作、奖励……），彼此通过配置类组合在一起，不需要手写仿真循环。

参考模板：`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

主入口文件：`source/go2_demo/go2_demo/tasks/manager_based/go2_demo/go2_demo_velocity.py`

### 3.1 机器人资产配置

**文件**：`source/go2_demo/go2_demo/assets/robot/unitree.py`

这是整个项目最关键的配置文件之一。它告诉 IsaacLab 怎么把机器人模型加载进仿真，以及关节用什么方式驱动。如果是自己购买的机器人，需要向厂商索取 URDF/USD 文件和真实的 PD 参数。

```python
UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",
        activate_contact_sensors=True,   # 开启接触力传感，终止条件和奖励函数都需要
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,      # 关闭自碰撞，加速仿真
            solver_position_iteration_count=4,  # 位置求解器迭代次数，影响稳定性
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),      # 初始高度 0.4 m，刚好让脚接触地面
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint":  0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
    ),
    soft_joint_pos_limit_factor=0.9,  # 以 90% 的关节限位作为软限位，防止撞硬限位
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,     # 最大输出力矩 (N·m)
            velocity_limit=30.0,   # 最大关节速度 (rad/s)
            stiffness=25.0,        # PD 控制中的 P 增益
            damping=0.5,           # PD 控制中的 D 增益
            friction=0.01,         # 执行器内摩擦系数
        ),
    },
)
```

**关节驱动模型选择：**

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `ImplicitActuatorCfg` | PhysX 内置，忽略电机动力学 | 快速原型验证 |
| `IdealPDActuatorCfg` | 理想 PD 控制，无力矩饱和 | 轻载机器人 |
| `DCMotorCfg` | 含力矩饱和的直流电机模型 | **真实腿足机器人（推荐）** |

> **注意**：`stiffness` 和 `damping` 直接影响机器人在仿真里的行为是否与真实一致，需要通过 `random_agent.py` 测试反复校对。

### 3.2 场景配置

**位置**：`go2_demo_velocity.py`，`RobotSceneCfg` 类

```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    # 超平坦地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,           # -1 表示只和同组内的物体碰撞
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # 机器人本体（ENV_REGEX_NS 是 IsaacLab 的多环境命名规则，
    # 会自动展开为 /World/envs/env_0/Robot, /World/envs/env_1/Robot ...）
    robot: ArticulationCfg = RobotCFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度传感器：从机器人头顶 20 m 向下射线，扫描脚下地形
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
    )

    # 接触力传感器：覆盖全身所有刚体，用于检测脚是否着地、身体是否碰撞
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,       # 记录最近 3 帧的接触力，用于计算 air time
        track_air_time=True,    # 追踪各腿的腾空时间，奖励函数用到
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, ...),
    )
```

> **多地形训练**：将 `terrain_type` 改为 `"generator"`，并配置 `TerrainGeneratorCfg`，可生成具有不同难度的崎岖地形，配合课程学习（Curriculum）使用效果更好。

### 3.3 观测空间

**位置**：`go2_demo_velocity.py`，`ObservationsCfg` 类

PPO 的 Actor-Critic 架构中，策略网络（Actor）和价值网络（Critic）可以接收不同的观测。这里两者看到的信息相同，实际项目里 Critic 通常会额外加入特权信息（如精确地形高度）来帮助学习。

```python
class PolicyCfg(ObsGroup):
    base_lin_vel      = ObsTerm(func=mdp.base_lin_vel, ...)      # 机体线速度 (机器人坐标系)  [3]
    base_ang_vel      = ObsTerm(func=mdp.base_ang_vel, ...)      # 机体角速度 (机器人坐标系)  [3]
    projected_gravity = ObsTerm(func=mdp.projected_gravity, ...) # 重力投影，表征姿态倾斜     [3]
    velocity_commands = ObsTerm(func=mdp.generated_commands, ...) # 目标速度指令              [3]
    joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel, ...)     # 相对默认位置的关节角偏差  [12]
    joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel, ...)     # 相对默认速度的关节速度偏差 [12]
    joint_effort      = ObsTerm(func=mdp.joint_effort, ...)      # 各关节输出力矩            [12]
    last_action       = ObsTerm(func=mdp.last_action, ...)       # 上一时刻策略输出的动作    [12]
    height_scanner    = ObsTerm(func=mdp.height_scan, ...)       # 脚下地形高度扫描          [N]

    def __post_init__(self):
        self.enable_corruption = True   # 开启噪声注入，增强 Sim-to-Real 鲁棒性
        self.concatenate_terms = True   # 把所有观测项拼成一个向量送给网络
```

**各观测项说明：**

| 观测项 | 维度 | 作用 |
|--------|------|------|
| `base_lin_vel` | 3 | 让策略知道机器人当前速度，与指令对比计算追踪误差 |
| `base_ang_vel` | 3 | 转向控制 |
| `projected_gravity` | 3 | 姿态感知，防止摔倒 |
| `velocity_commands` | 3 | 告诉策略「该往哪走」，相当于目标输入 |
| `joint_pos_rel` | 12 | 关节当前角度，策略据此规划下一步动作 |
| `joint_vel_rel` | 12 | 关节运动趋势 |
| `joint_effort` | 12 | 力矩反馈，辅助策略学习省力步态 |
| `last_action` | 12 | 提供时序信息，有助于学到平滑动作 |
| `height_scanner` | N | 感知脚下地形，为越障预留接口 |

所有项都配有 `noise`（均匀噪声）和 `clip`（数值截断），以模拟真实传感器噪声并防止数值爆炸。

> **CriticCfg** 的观测项与 PolicyCfg 完全相同，但不需要 `__post_init__` 方法（Critic 通常不加噪声，这里为保持一致性也保留了噪声，可视需求调整）。

### 3.4 动作空间

**位置**：`go2_demo_velocity.py`，`ActionsCfg` 类

```python
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],        # 控制所有 12 个关节
        scale={
            ".*_hip_joint":   0.125,   # 髋关节动作范围较小
            ".*_thigh_joint": 0.25,
            ".*_calf_joint":  0.25,
        },
        use_default_offset=True,   # 网络输出的是相对默认关节角的增量，而非绝对值
        clip={".*": (-100.0, 100.0)},
    )
```

策略网络输出的是 12 维向量（对应 Go2 的 12 个关节），经过 `scale` 缩放后加到初始关节角上，得到目标关节角，再交给 PD 控制器执行。`use_default_offset=True` 让网络只需学习「偏差量」，而不用学绝对位置，降低了学习难度。

### 3.5 速度指令

**位置**：`go2_demo_velocity.py`，`CommandsCfg` 类

```python
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # 每 10 秒重新采样一次目标速度
        rel_standing_envs=0.02,              # 2% 的并行环境会收到零速指令（站立任务）
        heading_command=True,                # 开启航向角控制
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.2),   # 前进速度范围 m/s
            lin_vel_y=(0.0, 0.0),   # 侧向速度（当前任务不启用）
            ang_vel_z=(-1.0, 1.0),  # 转向角速度范围 rad/s
        ),
        # 课程学习的速度上限，策略成熟后才会逐步开放
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 6.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
```

训练过程中，指令管理器（Command Manager）在每个 episode 里持续为机器人生成随机速度目标。机器人的任务就是让实际速度尽量贴近这个目标。

### 3.6 奖励函数

**位置**：`go2_demo_velocity.py`，`RewardsCfg` 类

奖励函数的设计直接决定训练出来的步态质量。这里采用常见的「正向激励 + 惩罚约束」结构，同时引入了 **AER（Adaptive Efficiency Reward）** 机制（见 `aer_env.py`）：

```
总奖励 = 正向奖励 × exp(惩罚奖励 / σ)
```

这种设计让正向奖励与惩罚奖励相乘，避免机器人为追求正奖励而忽视惩罚，也防止惩罚过大导致机器人选择静止不动。

**正向激励（越高越好）：**

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `track_lin_vel_xy` | +1.0 | 线速度追踪，核心任务奖励 |
| `track_ang_vel_z` | +0.5 | 转向角速度追踪 |
| `energy_new_actual` | +0.8 | 能量效率奖励，鼓励机器人用省力的步态行走 |

**惩罚约束（越小越好）：**

| 惩罚项 | 权重 | 说明 |
|--------|------|------|
| `flat_orientation_l2` | -5.0 | 惩罚机身倾斜，保持平衡 |
| `base_height` | -30.0 | 维持目标站立高度 (0.3 m) |
| `undesired_contacts` | -5.0 | 防止大腿、小腿蹭地 |
| `dof_pos_limits` | -10.0 | 避免关节超软限位 |
| `feet_slip` | -0.04 | 惩罚脚部滑移 |
| `action_rate` | -0.01 | 惩罚相邻时刻动作突变，使控制更平滑 |
| `joint_torques` | -1e-4 | 惩罚力矩过大，节省能量 |
| `joint_vel` | -1e-4 | 惩罚关节速度过大 |
| `joint_acc` | -2.5e-7 | 惩罚关节加速度过大，减少机械冲击 |
| `base_linear_velocity` | -0.05 | 惩罚 Z 轴方向（垂直）的线速度抖动 |
| `base_angular_velocity` | -0.001 | 惩罚 XY 轴方向的翻滚/俯仰角速度 |

> **常见问题**：如果训练后机器人站着不动，说明惩罚权重相对于正向奖励过大。首先检查 `track_lin_vel_xy` 的权重是否足够，然后适当降低 `flat_orientation_l2` 或 `base_height` 的惩罚强度。

### 3.7 终止条件

**位置**：`go2_demo_velocity.py`，`TerminationsCfg` 类

```python
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 机身接触地面（摔倒），需要 ContactSensorCfg 支持
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # 机器人倾斜超过阈值（约 46°）
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
```

每个 episode 时长为 20 秒（`episode_length_s = 20.0`）。如果机器人提前摔倒，episode 提早结束，拿不到后续奖励，这本身就是一个自然的惩罚信号。

### 3.8 域随机化

**位置**：`go2_demo_velocity.py`，`EventCfg` 类

域随机化（Domain Randomization）的作用是缩小 Sim-to-Real Gap，让在仿真里训练的策略能直接部署到真实机器人上。

IsaacLab 把事件分为三类触发时机：

| 触发模式 | 时机 | 典型用途 |
|---------|------|---------|
| `startup` | 仿真启动后只执行一次 | 随机化机器人外观、质量属性 |
| `reset` | 每次 episode 结束重置时 | 随机化初始位姿、初始速度 |
| `interval` | 按固定间隔周期执行 | 模拟外部推力扰动、随机关节摩擦 |

本项目目前只配置了 `reset` 事件——每次重置时在 XY 平面随机摆放机器人位置和朝向：

```python
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0), ...},
        },
    )
```

**建议后续扩展的随机化项目：**

- `startup`：随机化机器人各连杆质量（±10%）、转动惯量
- `reset`：随机化关节初始角度
- `interval`：施加随机推力（模拟被人踢一脚），随机化 PD 增益

### 3.9 课程学习

**位置**：`go2_demo_velocity.py`，`CurriculumCfg` 类

```python
class CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
```

课程学习（Curriculum Learning）让任务难度随着训练进展逐渐提升。这里的 `lin_vel_cmd_levels` 会根据最近的速度追踪奖励动态调整速度指令的采样范围，从较低的速度开始，逐步放开到 `limit_ranges` 中配置的最高速度（6 m/s）。

如果跳过课程学习，直接让机器人学习 6 m/s 的高速奔跑，训练会非常不稳定。

### 3.10 注册 Gym 环境

**文件**：`source/go2_demo/go2_demo/tasks/manager_based/go2_demo/__init__.py`

```python
gym.register(
    id="Go2-velocity-v0",
    # 如果没有重写 ManagerBasedRLEnv，这里填 "isaaclab.envs:ManagerBasedRLEnv"
    # 本项目重写了奖励计算逻辑（AER），所以指向自定义类
    entry_point=f"{__name__}.aer_env:AERManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 指定环境配置类
        "env_cfg_entry_point": f"{__name__}.go2_demo_velocity:GO2RobotDemoEnv",
        # 指定 PPO 训练超参数
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
```

`id` 就是后续所有训练和测试命令中 `--task` 参数的值。

**关于 `AERManagerBasedRLEnv`**：这是对 `ManagerBasedRLEnv` 的一次扩展，主要修改了奖励计算逻辑——把正向奖励和负向惩罚分开累计，再按指数公式合并，从而避免过大的惩罚项「淹没」任务奖励。

### 3.11 PPO 训练超参数

**文件**：`source/go2_demo/go2_demo/tasks/manager_based/go2_demo/agents/rsl_rl_ppo_cfg.py`

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env  = 24          # 每次更新前每个环境收集多少步数据
    max_iterations     = 20000       # 最大训练轮数
    save_interval      = 1000        # 每隔多少轮保存一次模型
    experiment_name    = "go2_demo"  # 日志和模型的保存目录名

    policy = RslRlPpoActorCriticCfg(
        init_noise_std        = 1.0,               # 初始动作噪声标准差（越大探索越充分）
        actor_hidden_dims     = [512, 256, 128],   # Actor 网络结构
        critic_hidden_dims    = [512, 256, 128],   # Critic 网络结构
        activation            = "elu",             # 激活函数
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef       = 1.0,    # 价值函数损失系数
        use_clipped_value_loss = True,
        clip_param            = 0.2,    # PPO clip 系数，控制每次更新步幅
        entropy_coef          = 0.01,   # 熵正则化系数，鼓励探索
        num_learning_epochs   = 5,      # 每批数据执行多少轮梯度更新
        num_mini_batches      = 4,      # 每轮拆成几个 mini-batch
        learning_rate         = 1.0e-3,
        schedule              = "adaptive",  # 根据 KL 散度自适应调整学习率
        gamma                 = 0.99,   # 折扣因子
        lam                   = 0.95,   # GAE lambda，越大越依赖远期奖励
        desired_kl            = 0.01,   # 期望 KL 散度（自适应学习率的目标）
        max_grad_norm         = 1.0,    # 梯度裁剪
    )
```

**有效数据量 = num_envs × num_steps_per_env**。4096 个并行环境 × 24 步 = 每轮近 10 万条样本，这也是为什么 IsaacLab 能快速训练腿足机器人的原因。

主环境配置中的关键时序参数：

```python
self.decimation = 4          # 策略每 4 个物理步执行一次（控制频率 = 1 / (4 × 0.005) = 50 Hz）
self.sim.dt = 0.005          # 物理仿真步长 5 ms（仿真频率 200 Hz）
self.episode_length_s = 20.0 # 每个 episode 持续 20 秒
```

---

## 4. 训练与测试

### 验证环境

先用零动作测试环境是否能正常启动（机器人不会移动）：

```bash
python scripts/zero_agent.py --task Go2-velocity-v0 --num_envs 16
```

再用随机动作测试机器人响应（机器人会乱动）：

```bash
python scripts/random_agent.py --task Go2-velocity-v0 --num_envs 16
```

观察机器人状态是否异常（例如肢体穿透地面、关节疯狂抖动）。如有异常，一般是 `unitree.py` 里 PD 参数（`stiffness`、`damping`）设置不合适——`stiffness` 过大会导致抖动，过小会导致关节无力。

### 开始训练

```bash
# 无头模式训练（推荐），附带录制训练视频
python scripts/rsl_rl/train.py \
    --task Go2-velocity-v0 \
    --num_envs 4096 \
    --max_iterations 50000 \
    --headless \
    --video

# 如果需要实时查看仿真画面（会显著降低训练速度）
python scripts/rsl_rl/train.py \
    --task Go2-velocity-v0 \
    --num_envs 512
```

训练产物保存在 `logs/rsl_rl/go2_demo/<timestamp>/` 目录下。

### 查看训练曲线

```bash
tensorboard --logdir=logs/rsl_rl/go2_demo/<timestamp>/
```

重点关注这几条曲线：

- **Episode Reward**：整体是否在上升
- **track_lin_vel_xy / track_ang_vel_z**：速度追踪奖励，这两条曲线是任务的核心指标
- **KL Divergence**：应该稳定在 `desired_kl=0.01` 附近，持续偏大说明学习不稳定

### 回放训练好的策略

```bash
python scripts/rsl_rl/play.py --task Go2-velocity-v0 --num_envs 16
```

同时会在 `logs/rsl_rl/go2_demo/<timestamp>/exported/` 下导出 ONNX 或 JIT 格式的模型，可用于部署到真实机器人。

---

## 5. 调试技巧

**机器人站着不动**
策略学到了「静止也不会被终止」的偷懒方案。首先提高速度追踪奖励权重（`track_lin_vel_xy` 和 `track_ang_vel_z`），或排查是否某个惩罚项权重过大。

**机器人疯狂抖动**
通常是 `stiffness` 过大，或者 `action_rate` 惩罚权重太小。降低 `stiffness` 或增大 `action_rate` 的惩罚权重（如从 -0.01 增加到 -0.05）。

**训练崩溃（奖励突然大幅下降）**
可能是学习率过大或梯度爆炸。检查 `max_grad_norm`，适当降低 `learning_rate`，或者把 `schedule` 改为固定学习率后逐步排查。

**Sim-to-Real 迁移效果差**
加强域随机化，特别是关节摩擦、质量属性和外部扰动力。同时检查真实机器人的 PD 参数是否与仿真配置一致。

**模型路径问题**
`unitree.py` 中的 `UNITREE_MODEL_DIR` 使用的是绝对路径，换机器后记得修改，或改用相对路径：

```python
import os
UNITREE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "unitree_model")
```

---

## 参考资料

- [IsaacLab 官方文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab)
- [IsaacLab GitHub](https://github.com/isaac-sim/IsaacLab)
- [宇树机器人开源模型（HuggingFace）](https://huggingface.co/datasets/unitreerobotics/unitree_model)
- [RSL-RL PPO 实现](https://github.com/leggedrobotics/rsl_rl)
