"""
扑翼优化项目的配置文件。
包含物理仿真、强化学习训练和可视化所需的所有参数。
"""

import numpy as np

# 物理参数
PHYSICS_CONFIG = {
    'fluid_density': 1.225,    # kg/m^3，流体密度
    'chord_length': 1.0,       # m，弦长
    # 'wing_span': 0.3,          # m，翼展
    'max_angle': 1.5,      # rad，最大俯仰角
    'max_displacement': 2.6,   # m，最大位移
    # 'max_velocity': 2.0,      # m/s，最大速度
    'max_velocity_increment': 0.4,  # m/s，最大速度增量
    'dt': 0.1,               # s，时间步长
    'action_interval': 0.1,    # s，动作更新间隔
    'simulation_time': 30.0,   # s，每回合总仿真时间
    # 'cfd_steps': 1,          # 每个RL步骤的CFD计算步数
    # 'adjustment_time': 0,   # s，速度平滑过渡时间
    'C_t_set': 40.0,          # 基线推力系数
    'fluid_velocity': 1.0,     # m/s，来流速度

    # 奖励函数参数
    'pitch_penalty_weight': 1.0,  # 俯仰角惩罚权重 beta_alpha
    'heave_penalty_weight': 1.0,  # 升沉位移惩罚权重 beta_y
    'inline_penalty_weight': 1.0, # 轴向位移惩罚权重 beta_x
}

# 训练参数
TRAINING_CONFIG = {
    # 'seed': 42,                  # 随机种子
    'hidden_dim': 512,           # 隐藏层维度
    'num_episodes': 500,          # 训练回合数
    # 'max_steps_per_episode': PHYSICS_CONFIG['simulation_time'] / PHYSICS_CONFIG['action_interval'],  # 每回合最大动作步数 (30.0秒/0.05秒)
    'learning_rate': 1e-4,       # 学习率
    'gamma': 0.99,               # 折扣因子
    'clip_epsilon': 0.2,         # PPO裁剪范围
    'gae_lambda': 0.95,          # GAE参数
    'update_epochs': 10,         # 每次更新的训练轮数
    'mini_batch_size': 128,       # 小批量大小
    'max_grad_norm': 1.0,        # 梯度裁剪范围
    'entropy_coef': 0.05,        # 熵正则化系数
    'value_coef': 0.5,          # 价值函数系数
}

# 环境参数
ENV_CONFIG = {
    'dof_mode': '2dof',        # 自由度模式：1dof/2dof/3dof
    'reward_weights': {
        'thrust_coef': 1.0,    # 推力系数权重
        'stability': 0.5,      # 稳定性权重
        'energy': 0.3,         # 能耗权重
    },
    # 'max_steps_per_episode': 1000,  # 每回合最大步数
    'physics_config': PHYSICS_CONFIG,  # 物理参数
    'cfd_zmq': {
        'host': '127.0.0.1',
        'port': 5555,
        'wing_count': 2,       # 翼数量：1=单翼，2=双翼；需与 CFD 服务端一致
        'timeout_ms': 600000,
        # 每个回合 RESET 后先用零速度增量预推进的物理时间；不计入RL回合步数
        'warmup_time': 2.0,
        # dof: PPO动作维度随 dof_mode 变化，但发给CFD时始终补齐6个协议字段
        # all: PPO动作直接为 [w1_domega,w1_dsau,w1_dsbv,w2_domega,w2_dsau,w2_dsbv]
        'action_mode': 'dof',
        # dof_mode=2dof 时，选择 sb 或 sa 作为升沉自由度，另一个自由度速度增量补0
        'motion_axis_2dof': 'sb',
        'flow_save': {
            'enabled': True,
            # 例如 enabled=True, episode_interval=10, step_interval=20 表示每10个回合每20步保存一次流场
            'episode_interval': 1,
            'step_interval': 0,
            # 也可以显式指定回合/步数列表，满足任一调度条件即SAVE
            'episodes': [100, 150,  200, 250, 300, 350, 400, 450, 500, 550, 600],
            'steps': [],
            # 按正式运动时间保存，单位秒；不包含 warmup_time，例如 [5.0] 表示正式运动第5秒
            'times': [5.0, 10.0, 15.0, 20.0, 25.0],
            # 若设置为数值，则 total_fx,total_fy,total_m 的欧氏变化超过阈值时触发SAVE
            'force_delta_threshold': None,
            # SAVE收到ERROR时，挂起到后续STEP成功后重试，最多重试次数
            'max_retries': 3,
        },
    },
}

# 可视化参数
VIS_CONFIG = {
    'plot_interval': 10,       # 每N步更新一次图像
    'save_plots': True,        # 是否保存图像
    'save_animations': False,   # 是否保存动画
    'fps': 30,                # 动画帧率
}
