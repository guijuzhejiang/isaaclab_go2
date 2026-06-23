# Go2 机器狗速度跟随训练项目

[English Version](./README.md)

本项目基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 框架，实现宇树 Go2 机器人在平坦地面上的 PPO 强化学习速度跟随训练。

## 核心功能

- **隔离开发**：作为 IsaacLab 的外部项目，独立于核心仓库。
- **Manager-Based 框架**：使用 IsaacLab 最新的配置驱动式环境开发模式。
- **AER 奖励机制**：引入 Adaptive Efficiency Reward，优化正向激励与惩罚的整合逻辑。
- **课程学习**：支持速度指令的动态课程升级。

## 快速上手

### 1. 环境准备

- 确保已安装 IsaacLab。推荐使用 conda 环境 `py312_cu121`。
- 克隆本项目并安装：

```bash
conda activate py312_cu121
python -m pip install -e source/go2_demo
```

### 2. 准备机器人模型

从 [HuggingFace](https://huggingface.co/datasets/unitreerobotics/unitree_model) 下载 Go2 模型，放入 `source/go2_demo/go2_demo/assets/robot/unitree_model`。

### 3. 运行训练与测试

**验证环境：**
```bash
# 零动作测试
python scripts/zero_agent.py --task Go2-velocity-v0 --num_envs 16
# 随机动作测试
python scripts/random_agent.py --task Go2-velocity-v0 --num_envs 16
```

**开始训练（无头模式）：**
```bash
python scripts/rsl_rl/train.py --task Go2-velocity-v0 --num_envs 4096 --headless --video
```

**回放策略：**
```bash
python scripts/rsl_rl/play.py --task Go2-velocity-v0 --num_envs 16
```

## 详细文档

更多细节请参考 `doc/` 目录：
- [训练指南](./doc/go2_demo_guide.md)：详细的环境配置、项目结构和调试技巧。
- [MDP 扩展指南](./doc/go2_demo_custom_mdp.md)：自定义奖励函数、课程学习和指令扩展的实现说明。
