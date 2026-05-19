"""
扑翼系统的运动学模型。
处理不同自由度模式下的运动更新和约束。
"""
import numpy as np
from config.config import PHYSICS_CONFIG

class KinematicsModel:
    def __init__(self, dof_mode='3dof'):
        self.dof_mode = dof_mode
        self.max_angle = PHYSICS_CONFIG['max_angle']
        self.max_displacement = PHYSICS_CONFIG['max_displacement']
        self.max_velocity_increment = PHYSICS_CONFIG['max_velocity_increment']
        self.dt = PHYSICS_CONFIG['dt']
        self.action_interval = PHYSICS_CONFIG['action_interval']
        self.steps_per_action = int(self.action_interval / self.dt)
        self.current_time = 0.0

        # 根据自由度模式初始化状态维度
        if self.dof_mode == '1dof':
            self.n_x_dim = 0  # 没有x方向维度
            self.n_wing_dim = 1    # 每个翼一个俯仰自由度
        elif self.dof_mode == '2dof':
            self.n_x_dim = 0  # 没有x方向维度
            self.n_wing_dim = 2    # 每个翼两个自由度(y, theta)
        else:  # 3dof
            self.n_x_dim = 1  # x维度
            self.n_wing_dim = 2    # 每个翼两个独立自由度(y, theta)

        self.n_total_dim = self.n_x_dim + 2 * self.n_wing_dim  # 总维度

        # x方向状态初始化
        self.x_position = np.zeros(self.n_x_dim)      # [x] 或 []
        self.x_velocity = np.zeros(self.n_x_dim)      # [vx] 或 []
        self.x_acceleration = np.zeros(self.n_x_dim)  # [ax] 或 []

        # 前翼状态
        self.wing1_position = np.zeros(self.n_wing_dim)      # [theta] 或 [y, theta] 或 [y, theta]
        self.wing1_velocity = np.zeros(self.n_wing_dim)      # [omega] 或 [vy, omega] 或 [vy, omega]
        self.wing1_acceleration = np.zeros(self.n_wing_dim)  # [alpha] 或 [ay, alpha] 或 [ay, alpha]

        # 后翼状态
        self.wing2_position = np.zeros(self.n_wing_dim)      # [theta] 或 [y, theta] 或 [y, theta]
        self.wing2_velocity = np.zeros(self.n_wing_dim)      # [omega] 或 [vy, omega] 或 [vy, omega]
        self.wing2_acceleration = np.zeros(self.n_wing_dim)  # [alpha] 或 [ay, alpha] 或 [ay, alpha]

        # 存储历史状态
        self.prev_x_position = np.zeros(self.n_x_dim)
        self.prev_x_velocity = np.zeros(self.n_x_dim)
        self.prev_x_acceleration = np.zeros(self.n_x_dim)

        self.prev_wing1_position = np.zeros(self.n_wing_dim)
        self.prev_wing1_velocity = np.zeros(self.n_wing_dim)
        self.prev_wing1_acceleration = np.zeros(self.n_wing_dim)

        self.prev_wing2_position = np.zeros(self.n_wing_dim)
        self.prev_wing2_velocity = np.zeros(self.n_wing_dim)
        self.prev_wing2_acceleration = np.zeros(self.n_wing_dim)

    def step(self, action):
        """执行一步运动更新"""
        # 保存当前状态作为历史状态
        self.prev_x_position = self.x_position.copy()
        self.prev_x_velocity = self.x_velocity.copy()
        self.prev_x_acceleration = self.x_acceleration.copy()

        self.prev_wing1_position = self.wing1_position.copy()
        self.prev_wing1_velocity = self.wing1_velocity.copy()
        self.prev_wing1_acceleration = self.wing1_acceleration.copy()

        self.prev_wing2_position = self.wing2_position.copy()
        self.prev_wing2_velocity = self.wing2_velocity.copy()
        self.prev_wing2_acceleration = self.wing2_acceleration.copy()

        # 分解动作向量
        action = np.array(action).reshape(-1)
        if self.dof_mode == '1dof':
            x_action = np.array([])  # 空
            wing1_action = action[0:1]    # [omega1]
            wing2_action = action[1:2]    # [omega2]
        elif self.dof_mode == '2dof':
            x_action = np.array([])  # 空
            wing1_action = action[0:2]    # [ay1, omega1]
            wing2_action = action[2:4]    # [ay2, omega2]
        else:  # 3dof
            x_action = action[0:1]  # [ax]
            wing1_action = action[1:3]   # [ay1, omega1]
            wing2_action = action[3:5]   # [ay2, omega2]

        # 处理x方向的运动
        if self.n_x_dim > 0:
            if np.any(np.abs(self.x_position) > self.max_displacement * 0.80):
                x_action = -np.sign(self.x_position)
            self.x_position += self.x_velocity * self.dt
            velocity_increments = x_action * self.max_velocity_increment * (self.dt / self.action_interval)
            self.x_velocity += velocity_increments
            self.x_acceleration = velocity_increments / self.dt

        # 处理前翼的运动
        self._update_wing_motion(
            self.wing1_position, self.wing1_velocity, self.wing1_acceleration,
            wing1_action
        )

        # 处理后翼的运动
        self._update_wing_motion(
            self.wing2_position, self.wing2_velocity, self.wing2_acceleration,
            wing2_action
        )

        # 更新时间
        self.current_time += self.dt

        # 应用运动约束
        is_valid = self._apply_constraints()

        return self.get_state_vector(), is_valid

    def _update_wing_motion(self, position, velocity, acceleration, action):
        """更新单个翼的运动状态"""
        # 检查边界并调整动作
        if self.dof_mode == '1dof':
            # 只有俯仰角约束
            if abs(position[0]) > self.max_angle * 0.80:
                action[0] = -np.sign(position[0])
        elif self.dof_mode == '2dof':
            # 检查升沉位移
            if abs(position[0]) > self.max_displacement * 0.80:
                action[0] = -np.sign(position[0])
            # 检查俯仰角
            if abs(position[1]) > self.max_angle * 0.80:
                action[1] = -np.sign(position[1])
        else:  # 3dof
            # 检查升沉位移
            if abs(position[0]) > self.max_displacement * 0.80:
                action[0] = -np.sign(position[0])
            # 检查俯仰角
            if abs(position[1]) > self.max_angle * 0.80:
                action[1] = -np.sign(position[1])

        # 计算速度增量并更新状态
        velocity_increments = action * self.max_velocity_increment * (self.dt / self.action_interval)
        position += velocity * self.dt
        velocity += velocity_increments
        acceleration[:] = velocity_increments / self.dt

    def _apply_constraints(self):
        """应用运动约束，返回是否在有效范围内"""
        is_valid = True

        # 检查x方向的约束
        if self.n_x_dim > 0:
            for i in range(self.n_x_dim):
                if abs(self.x_position[i]) > self.max_displacement:
                    self.x_position[i] = np.sign(self.x_position[i]) * self.max_displacement
                    self.x_velocity[i] = 0
                    self.x_acceleration[i] = -np.sign(self.x_position[i]) * self.max_velocity_increment / self.dt
                    is_valid = True

        # 检查前翼的约束
        is_valid &= self._apply_wing_constraints(
            self.wing1_position, self.wing1_velocity, self.wing1_acceleration
        )

        # 检查后翼的约束
        is_valid &= self._apply_wing_constraints(
            self.wing2_position, self.wing2_velocity, self.wing2_acceleration
        )

        return is_valid

    def _apply_wing_constraints(self, position, velocity, acceleration):
        """应用单个翼的运动约束"""
        is_valid = True

        if self.dof_mode == '1dof':
            # 只检查俯仰角约束
            if abs(position[0]) > self.max_angle:
                position[0] = np.sign(position[0]) * self.max_angle
                velocity[0] = 0
                acceleration[0] = -np.sign(position[0]) * self.max_velocity_increment / self.dt
                is_valid = True
        elif self.dof_mode == '2dof':
            # 检查升沉位移约束
            if abs(position[0]) > self.max_displacement:
                position[0] = np.sign(position[0]) * self.max_displacement
                velocity[0] = 0
                acceleration[0] = -np.sign(position[0]) * self.max_velocity_increment / self.dt
                is_valid = True
            # 检查俯仰角约束
            if abs(position[1]) > self.max_angle:
                position[1] = np.sign(position[1]) * self.max_angle
                velocity[1] = 0
                acceleration[1] = -np.sign(position[1]) * self.max_velocity_increment / self.dt
                is_valid = True
        else:  # 3dof
            # 检查升沉位移约束
            if abs(position[0]) > self.max_displacement:
                position[0] = np.sign(position[0]) * self.max_displacement
                velocity[0] = 0
                acceleration[0] = -np.sign(position[0]) * self.max_velocity_increment / self.dt
                is_valid = True
            # 检查俯仰角约束
            if abs(position[1]) > self.max_angle:
                position[1] = np.sign(position[1]) * self.max_angle
                velocity[1] = 0
                acceleration[1] = -np.sign(position[1]) * self.max_velocity_increment / self.dt
                is_valid = True

        return is_valid

    def reset(self):
        """重置运动状态"""
        self.x_position = np.zeros(self.n_x_dim)
        self.x_velocity = np.zeros(self.n_x_dim)
        self.x_acceleration = np.zeros(self.n_x_dim)

        self.wing1_position = np.zeros(self.n_wing_dim)
        self.wing1_velocity = np.zeros(self.n_wing_dim)
        self.wing1_acceleration = np.zeros(self.n_wing_dim)

        self.wing2_position = np.zeros(self.n_wing_dim)
        self.wing2_velocity = np.zeros(self.n_wing_dim)
        self.wing2_acceleration = np.zeros(self.n_wing_dim)

        self.prev_x_position = np.zeros(self.n_x_dim)
        self.prev_x_velocity = np.zeros(self.n_x_dim)
        self.prev_x_acceleration = np.zeros(self.n_x_dim)

        self.prev_wing1_position = np.zeros(self.n_wing_dim)
        self.prev_wing1_velocity = np.zeros(self.n_wing_dim)
        self.prev_wing1_acceleration = np.zeros(self.n_wing_dim)

        self.prev_wing2_position = np.zeros(self.n_wing_dim)
        self.prev_wing2_velocity = np.zeros(self.n_wing_dim)
        self.prev_wing2_acceleration = np.zeros(self.n_wing_dim)

        self.current_time = 0.0
        return self.get_state_vector()

    def get_state_vector(self):
        """获取当前状态向量"""
        current_state = []
        history_state = []

        # 添加当前状态
        if self.n_x_dim > 0:  # 3DOF情况，存储格式：[x, y1, theta1, y2, theta2, vx, vy1, omega1, vy2, omega2, ax, ay1, alpha1, ay2, alpha2]
            current_state.extend([
                self.x_position,
                self.wing1_position,
                self.wing2_position,
                self.x_velocity,
                self.wing1_velocity,
                self.wing2_velocity,
                self.x_acceleration,
                self.wing1_acceleration,
                self.wing2_acceleration
            ])
        else:  # 1DOF或2DOF情况,存储格式：[y1, theta1, y2, theta2, vy1, omega1, vy2, omega2, ay1, alpha1, ay2, alpha2]或[theta1, theta2, omega1, omega2, alpha1, alpha2]
            current_state.extend([
                self.wing1_position,
                self.wing2_position,
                self.wing1_velocity,
                self.wing2_velocity,
                self.wing1_acceleration,
                self.wing2_acceleration
            ])

        # 添加历史状态
        if self.n_x_dim > 0:  # 3DOF情况
            history_state.extend([
                self.prev_x_position,
                self.prev_wing1_position,
                self.prev_wing2_position,
                self.prev_x_velocity,
                self.prev_wing1_velocity,
                self.prev_wing2_velocity,
                self.prev_x_acceleration,
                self.prev_wing1_acceleration,
                self.prev_wing2_acceleration
            ])
        else:  # 1DOF或2DOF情况
            history_state.extend([
                self.prev_wing1_position,
                self.prev_wing2_position,
                self.prev_wing1_velocity,
                self.prev_wing2_velocity,
                self.prev_wing1_acceleration,
                self.prev_wing2_acceleration
            ])

        return np.concatenate([np.concatenate(current_state), np.concatenate(history_state)])
