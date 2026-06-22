import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


UNITREE_MODEL_DIR = '/home/zzg/workspace/pycharm/go2_demo/source/go2_demo/go2_demo/assets/robot/unitree_model'

UNITREE_GO2_CFG = ArticulationCfg(
    # 定义加载方式
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd", # USD文件路径（URDF文件加载方式见官方文档）
        activate_contact_sensors=True, # 读取传感器信息
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    # 机器人的初始状态
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),    #初始状态下处于世界坐标系位置
        joint_pos={             #自身关节位置
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},  #关节速度
    ),
    # 软限位系数
    soft_joint_pos_limit_factor=0.9,
    # 核心：执行器模型
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5, # 输出力矩上限
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0, # PD控制中的P
            damping=0.5, # PD控制中的D
            friction=0.01, # 执行器摩擦系数
        ),
    },
)