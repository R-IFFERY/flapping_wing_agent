"""
基于OpenAI Gym接口实现的扑翼环境。
实现强化学习代理与物理仿真之间的交互。
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from physics.kinematics import KinematicsModel
from physics.aerodynamics import AerodynamicsModel
from .reward import RewardFunction
from config.config import PHYSICS_CONFIG, ENV_CONFIG
import os

class FlappingWingEnv(gym.Env):
    def __init__(self, dof_mode='1dof', use_cfd=False, save_dir=None):
        super().__init__()
        self.dof_mode = dof_mode
        self.use_cfd = use_cfd
        self.use_zmq_cfd = self.use_cfd
        self.zmq_config = ENV_CONFIG.get('cfd_zmq', {})
        self.zmq_action_mode = self.zmq_config.get('action_mode', 'dof')
        self.zmq_motion_axis_2dof = self.zmq_config.get('motion_axis_2dof', 'sb')
        self.zmq_warmup_time = float(self.zmq_config.get('warmup_time', 0.0) or 0.0)
        self.cfd_time_offset = 0.0
        self.max_steps = int(PHYSICS_CONFIG['simulation_time'] / PHYSICS_CONFIG['action_interval'])
        self.steps_per_action = int(PHYSICS_CONFIG['action_interval'] / PHYSICS_CONFIG['dt'])
        self.current_step = 0
        self.simulation_time = 0.0
        self.save_dir = save_dir
        self.current_episode = 0
        self.last_cl1 = 0
        self.last_cl2 = 0

        # 为气动数据创建专门的保存目录
        if self.save_dir:
            self.aero_save_dir = os.path.join(self.save_dir, 'aerodynamics_data')
            os.makedirs(self.aero_save_dir, exist_ok=True)
        else:
            self.aero_save_dir = None

        # 根据自由度模式确定状态和动作维度
        if dof_mode == '1dof':
            self.state_dim = 12  # 6(前翼) + 6(后翼)
            self.action_dim = 2   # 1(前翼角速度) + 1(后翼角速度)
        elif dof_mode == '2dof':
            self.state_dim = 24  # 12(前翼) + 12(后翼)
            self.action_dim = 4   # 2(前翼升沉+角速度) + 2(后翼升沉+角速度)
        else:  # 3dof
            self.state_dim = 30  # 3(共享) + 12(前翼) + 12(后翼) + 3(历史共享)
            self.action_dim = 5   # 1(共享轴向) + 2(前翼升沉+角速度) + 2(后翼升沉+角速度)

        if self.use_zmq_cfd and self.zmq_action_mode == 'all':
            self.action_dim = int(self.zmq_config.get('wing_count', 2)) * 3
        elif self.use_zmq_cfd and self.dof_mode == '3dof':
            self.action_dim = int(self.zmq_config.get('wing_count', 2)) * 3

        # 初始化物理模型
        self.kinematics = KinematicsModel(dof_mode)
        self.aerodynamics = AerodynamicsModel(use_cfd, save_dir=self.aero_save_dir)
        self.reward_function = RewardFunction()
        self.previous_cfd_current_state = np.zeros(self.state_dim // 2)

        # 定义动作空间
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # 添加可视化器的初始化
        self.visualizer = None  # 稍后设置可视化器

    def set_visualizer(self, visualizer):
        """设置可视化器"""
        self.visualizer = visualizer

    def step(self, action):
        """执行一个动作并推进环境"""
        if self.use_zmq_cfd:
            return self._step_zmq(action)

        self.current_step += 1
        rewards = []
        reached_boundary = False
        current_state = None
        stab = 0

        # 初始化空气动力相关的变量，防止在循环未执行或提前中断时未定义
        forces1 = forces2 = None
        cl1 = cl2 = ct1 = ct2 = cp1 = cp2 = pf = 0.0
        total_ct = total_cp = 0.0

        # 在一个动作更新间隔内执行多个物理时间步
        for i in range(self.steps_per_action):
            # 使用当前动作更新物理状态
            current_state, is_valid = self.kinematics.step(action)

            # 计算空气动力
            sim_time = self.simulation_time
            forces1, cl1, ct1, cp1, forces2, cl2, ct2, cp2, pf = self.aerodynamics.calculate_forces(self._get_wing1_state(current_state), self._get_wing2_state(current_state), sim_time)

            total_ct = ct1 + ct2
            total_cp = cp1 + cp2

            # 计算升力稳定性指标
            if sim_time > 0:
                stab = (cl1 + cl2 - self.last_cl1 - self.last_cl2)**2
            self.last_cl1 = cl1
            self.last_cl2 = cl2

            # 边界检查
            current_reward = 0
            if not is_valid:
                current_reward = -1000
                reached_boundary = True
            else:
                # 构建前后翼的运动参数
                motion_params = self._get_motion_params(current_state)
                # 计算综合奖励（考虑前后翼的贡献)
                current_reward = self.reward_function.calculate_reward(
                    pf, motion_params, action, stab)

            rewards.append(current_reward)
            self.simulation_time += PHYSICS_CONFIG['dt']

            if reached_boundary:
                break

        # 计算最终奖励
        reward = -1000 if reached_boundary else np.mean(rewards)

        # 更新可视化器
        if self.visualizer is not None:
            self.visualizer.update(current_state, None, reward, self.current_step, action, ct1, ct2, cp1, cp2, pf, stab)

        # 判断是否结束
        done = reached_boundary or self.current_step >= self.max_steps

        # 确定结束原因
        termination_reason = None
        if done:
            if reached_boundary:
                termination_reason = self._get_termination_reason(current_state)
            elif self.current_step >= self.max_steps:
                termination_reason = "达到最大步数"

        # 准备信息字典
        info = {
            'thrust_coefficient': total_ct,
            'wing1_ct': ct1,
            'wing2_ct': ct2,
            'wing1_cl': cl1,
            'wing2_cl': cl2,
            'wing1_cp': cp1,
            'wing2_cp': cp2,
            'power_factor': pf,
            'stable': stab,
            'wing1_forces': forces1,
            'wing2_forces': forces2,
            'is_valid': not reached_boundary,
            'simulation_time': self.simulation_time,
            'termination_reason': termination_reason,
            'boundary_violation': reached_boundary,
            'reward': reward
        }

        return current_state, reward, done, info

    def _step_zmq(self, action):
        """使用在线 CFD 返回的运动/气动力推进一个控制周期。"""
        self.current_step += 1
        action = np.array(action, dtype=float).reshape(-1)
        target_time = self._target_time_for_step(self.current_step)
        control_values = self._action_to_zmq_controls(action)

        result = self.aerodynamics.step_zmq(
            target_time=target_time,
            control_values=control_values,
            episode=self.current_episode,
            rl_step=self.current_step,
        )
        response = result["response"]
        current_state = self._state_from_zmq_response(response)
        self.simulation_time = response["time"]

        cl1 = result["cl1"]
        cl2 = result["cl2"]
        ct1 = result["ct1"]
        ct2 = result["ct2"]
        cp1 = result["cp1"]
        cp2 = result["cp2"]
        pf = result["pf"]
        total_ct = ct1 + ct2
        total_cp = cp1 + cp2
        forces1 = result["forces1"]
        forces2 = result["forces2"]

        stab = 0.0
        if self.current_step > 1:
            stab = (cl1 + cl2 - self.last_cl1 - self.last_cl2) ** 2
        self.last_cl1 = cl1
        self.last_cl2 = cl2

        motion_params = self._get_motion_params(current_state)
        reached_boundary = not self._is_zmq_motion_valid(response["motion_last"])
        reward = -1000 if reached_boundary else self.reward_function.calculate_reward(
            pf, motion_params, action, stab
        )

        if self.visualizer is not None:
            self.visualizer.update(current_state, None, reward, self.current_step, action, ct1, ct2, cp1, cp2, pf, stab)

        done = reached_boundary or self.current_step >= self.max_steps
        termination_reason = None
        if done:
            if reached_boundary:
                termination_reason = self._get_termination_reason(current_state)
            else:
                termination_reason = "达到最大步数"

        info = {
            'thrust_coefficient': total_ct,
            'wing1_ct': ct1,
            'wing2_ct': ct2,
            'wing1_cl': cl1,
            'wing2_cl': cl2,
            'wing1_cp': cp1,
            'wing2_cp': cp2,
            'power_factor': pf,
            'stable': stab,
            'wing1_forces': forces1,
            'wing2_forces': forces2,
            'total_power_coefficient': total_cp,
            'is_valid': not reached_boundary,
            'simulation_time': self.simulation_time,
            'termination_reason': termination_reason,
            'boundary_violation': reached_boundary,
            'reward': reward,
            'cfd_response': response,
            'flow_save_response': result.get("save_response"),
            'zmq_control_values': control_values,
        }
        return current_state, reward, done, info

    def _action_to_zmq_controls(self, action):
        scaled = np.clip(action, -1.0, 1.0) * PHYSICS_CONFIG['max_velocity_increment']
        if self.zmq_action_mode == 'all':
            expected = int(self.zmq_config.get('wing_count', 2)) * 3
            if len(scaled) != expected:
                raise ValueError(f"ZMQ all-DOF action length must be {expected}, got {len(scaled)}")
            return scaled.tolist()

        controls = np.zeros(6, dtype=float)
        if self.dof_mode == '1dof':
            if len(scaled) != 2:
                raise ValueError(f"1dof action length must be 2, got {len(scaled)}")
            controls[0] = scaled[0]
            controls[3] = scaled[1]
        elif self.dof_mode == '2dof':
            if len(scaled) != 4:
                raise ValueError(f"2dof action length must be 4, got {len(scaled)}")
            controls[0] = scaled[1]
            controls[3] = scaled[3]
            if self.zmq_motion_axis_2dof == 'sa':
                controls[1] = scaled[0]
                controls[4] = scaled[2]
            else:
                controls[2] = scaled[0]
                controls[5] = scaled[2]
        else:
            if len(scaled) == 6:
                controls[:] = scaled
            elif len(scaled) == 5:
                controls[0] = scaled[2]
                controls[1] = scaled[0]
                controls[2] = scaled[1]
                controls[3] = scaled[4]
                controls[4] = scaled[0]
                controls[5] = scaled[3]
            else:
                raise ValueError(f"3dof action length must be 6, got {len(scaled)}")
        return controls.tolist()

    def _target_time_for_step(self, step):
        target_time = self.cfd_time_offset + step * PHYSICS_CONFIG['action_interval']
        return round(round(target_time / 0.0001) * 0.0001, 10)

    def _run_zmq_warmup(self, initial_response):
        warmup_steps = self._warmup_steps()
        state = self._state_from_zmq_response(initial_response)
        if warmup_steps <= 0:
            self.cfd_time_offset = initial_response["time"]
            self.simulation_time = initial_response["time"]
            return state

        zero_controls = [0.0] * (int(self.zmq_config.get('wing_count', 2)) * 3)
        response = initial_response
        for k in range(1, warmup_steps + 1):
            target_time = round(round(k * PHYSICS_CONFIG['action_interval'] / 0.0001) * 0.0001, 10)
            result = self.aerodynamics.step_zmq(
                target_time=target_time,
                control_values=zero_controls,
                episode=self.current_episode,
                rl_step=0,
                record=False,
                allow_flow_save=False,
            )
            response = result["response"]
            state = self._state_from_zmq_response(response)

        self.cfd_time_offset = response["time"]
        self.simulation_time = response["time"]
        return state

    def _warmup_steps(self):
        if self.zmq_warmup_time <= 0:
            return 0
        raw_steps = self.zmq_warmup_time / PHYSICS_CONFIG['action_interval']
        steps = int(round(raw_steps))
        if not np.isclose(raw_steps, steps, rtol=0.0, atol=1e-9):
            raise ValueError("cfd_zmq.warmup_time must be an integer multiple of action_interval")
        return steps

    def _state_from_zmq_response(self, response):
        motion = response["motion_last"]
        current = self._current_state_from_zmq_motion(motion)
        state = np.concatenate([current, self.previous_cfd_current_state])
        self.previous_cfd_current_state = current.copy()
        return state.astype(np.float32)

    def _current_state_from_zmq_motion(self, motion):
        m1, m2 = motion[0], motion[1]
        if self.dof_mode == '1dof':
            return np.array([
                m1['angle'], m2['angle'],
                m1['omega'], m2['omega'],
                m1['omgac'], m2['omgac'],
            ], dtype=float)

        if self.dof_mode == '2dof':
            pos_key = self.zmq_motion_axis_2dof
            vel_key = 'sau' if pos_key == 'sa' else 'sbv'
            acc_key = 'saac' if pos_key == 'sa' else 'sbac'
            return np.array([
                m1[pos_key], m1['angle'], m2[pos_key], m2['angle'],
                m1[vel_key], m1['omega'], m2[vel_key], m2['omega'],
                m1[acc_key], m1['omgac'], m2[acc_key], m2['omgac'],
            ], dtype=float)

        shared_sa = 0.5 * (m1['sa'] + m2['sa'])
        shared_sau = 0.5 * (m1['sau'] + m2['sau'])
        shared_saac = 0.5 * (m1['saac'] + m2['saac'])
        return np.array([
            shared_sa, m1['sb'], m1['angle'], m2['sb'], m2['angle'],
            shared_sau, m1['sbv'], m1['omega'], m2['sbv'], m2['omega'],
            shared_saac, m1['sbac'], m1['omgac'], m2['sbac'], m2['omgac'],
        ], dtype=float)

    def _is_zmq_motion_valid(self, motion):
        for wing_motion in motion:
            if abs(wing_motion['angle']) > self.kinematics.max_angle:
                return False
            if abs(wing_motion['sa']) > self.kinematics.max_displacement:
                return False
            if abs(wing_motion['sb']) > self.kinematics.max_displacement:
                return False
        return True

    def _get_wing1_state(self, state):
        """从完整状态中提取前翼状态"""
        if self.dof_mode == '3dof':
            # 3DOF：[x, y1, theta1, vx, vy1, omega1, ax, ay1, alpha1]
            return np.array([
                state[0],   # x
                state[1],   # y1
                state[2],   # theta1
                state[5],   # vx
                state[6],   # vy1
                state[7],   # omega1
                state[10],  # ax
                state[11],  # ay1
                state[12]   # alpha1
            ])
        elif self.dof_mode == '2dof':
            # 2DOF 状态向量在 kinematics.get_state_vector 中的顺序为：
            # [y1, theta1, y2, theta2, vy1, omega1, vy2, omega2, ay1, alpha1, ay2, alpha2]
            # 因此前翼的完整子状态应为 [y1, theta1, vy1, omega1, ay1, alpha1]
            return np.array([
                state[0],  # y1
                state[1],  # theta1
                state[4],  # vy1
                state[5],  # omega1
                state[8],  # ay1
                state[9]   # alpha1
            ])
        else:  # 1DOF
            # 1DOF 状态向量顺序为：[theta1, theta2, omega1, omega2, alpha1, alpha2]
            # 前翼的子状态应为 [theta1, omega1, alpha1]
            return np.array([
                state[0],  # theta1
                state[2],  # omega1
                state[4]   # alpha1
            ])

    def _get_wing2_state(self, state):
        """从完整状态中提取后翼状态"""
        if self.dof_mode == '3dof':
            # 3DOF：[x, y2, theta2, vx, vy2, omega2, ax, ay2, alpha2]
            return np.array([
                state[0],   # x
                state[3],   # y2
                state[4],   # theta2
                state[5],   # vx
                state[8],   # vy2
                state[9],   # omega2
                state[10],  # ax
                state[13],  # ay2
                state[14]   # alpha2
            ])
        elif self.dof_mode == '2dof':
            # 2DOF 状态向量：
            # [y1, theta1, y2, theta2, vy1, omega1, vy2, omega2, ay1, alpha1, ay2, alpha2]
            # 后翼的完整子状态应为 [y2, theta2, vy2, omega2, ay2, alpha2]
            return np.array([
                state[2],  # y2
                state[3],  # theta2
                state[6],  # vy2
                state[7],  # omega2
                state[10], # ay2
                state[11]  # alpha2
            ])
        else:  # 1DOF
            # 1DOF 状态向量：[theta1, theta2, omega1, omega2, alpha1, alpha2]
            # 后翼子状态应为 [theta2, omega2, alpha2]
            return np.array([
                state[1],  # theta2
                state[3],  # omega2
                state[5]   # alpha2
            ])

    def _get_motion_params(self, state):
        """构建运动参数字典"""
        if self.dof_mode == '1dof':
            return {
                'wing1': {'pitch': state[0], 'heave': 0, 'inline': 0},
                'wing2': {'pitch': state[1], 'heave': 0, 'inline': 0}
            }
        elif self.dof_mode == '2dof':
            return {
                'wing1': {'pitch': state[1], 'heave': state[0], 'inline': 0},
                'wing2': {'pitch': state[3], 'heave': state[2], 'inline': 0}
            }
        else:  # 3dof
            return {
                'wing1': {'pitch': state[2], 'heave': state[1], 'inline': state[0]},
                'wing2': {'pitch': state[4], 'heave': state[3], 'inline': state[0]}
            }

    def _get_termination_reason(self, state):
        """获取终止原因的详细信息"""
        if self.dof_mode == '1dof':
            angle1, angle2 = abs(state[0]), abs(state[1])
            if angle1 > self.kinematics.max_angle or angle2 > self.kinematics.max_angle:
                return f"达到运动边界：前翼俯仰角 = {angle1:.2f} rad, 后翼俯仰角 = {angle2:.2f} rad (最大值：{self.kinematics.max_angle:.2f} rad)"
        elif self.dof_mode == '2dof':
            y1, theta1 = abs(state[0]), abs(state[1])
            y2, theta2 = abs(state[2]), abs(state[3])
            if max(theta1, theta2) > self.kinematics.max_angle:
                return f"达到运动边界：前翼俯仰角 = {theta1:.2f} rad, 后翼俯仰角 = {theta2:.2f} rad (最大值：{self.kinematics.max_angle:.2f} rad)"
            else:
                return f"达到运动边界：前翼升沉 = {y1:.2f} m, 后翼升沉 = {y2:.2f} m (最大值：{self.kinematics.max_displacement:.2f} m)"
        else:  # 3dof
            x = abs(state[0])
            y1, theta1 = abs(state[1]), abs(state[2])
            y2, theta2 = abs(state[3]), abs(state[4])
            if max(theta1, theta2) > self.kinematics.max_angle:
                return f"达到运动边界：前翼俯仰角 = {theta1:.2f} rad, 后翼俯仰角 = {theta2:.2f} rad (最大值：{self.kinematics.max_angle:.2f} rad)"
            elif max(y1, y2) > self.kinematics.max_displacement:
                return f"达到运动边界：前翼升沉 = {y1:.2f} m, 后翼升沉 = {y2:.2f} m (最大值：{self.kinematics.max_displacement:.2f} m)"
            else:
                return f"达到运动边界：轴向位移 = {x:.2f} m (最大值：{self.kinematics.max_displacement:.2f} m)"

    def reset(self):
        """重置环境到初始状态"""
        # 保存当前回合的数据
        self.aerodynamics.reset()  # 这会触发数据保存

        # 重置状态
        self.current_step = 0
        self.simulation_time = 0.0
        self.cfd_time_offset = 0.0
        self.last_cl1 = 0
        self.last_cl2 = 0
        self.previous_cfd_current_state = np.zeros(self.state_dim // 2)
        if self.use_zmq_cfd:
            response = self.aerodynamics.reset_zmq()
            return self._run_zmq_warmup(response)

        state = self.kinematics.reset()

        return state

    def set_episode(self, episode):
        """设置当前回合号"""
        self.current_episode = episode
        self.aerodynamics.set_episode(episode)

    def render(self, mode='human'):
        """渲染环境（由外部可视化器处理）"""
        pass

    def close(self):
        """关闭环境"""
        self.aerodynamics.close(stop_solver=self.use_zmq_cfd)
