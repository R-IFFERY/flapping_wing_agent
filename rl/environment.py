"""
基于OpenAI Gym接口实现的扑翼环境。
实现强化学习代理与物理仿真之间的交互。
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from physics.aerodynamics import AerodynamicsModel
from .reward import RewardFunction
from config.config import PHYSICS_CONFIG, ENV_CONFIG
import os

class FlappingWingEnv(gym.Env):
    def __init__(self, dof_mode='1dof', save_dir=None):
        super().__init__()
        self.dof_mode = dof_mode
        self.zmq_config = ENV_CONFIG.get('cfd_zmq', {})
        self.wing_count = int(self.zmq_config.get('wing_count', 2))
        self.zmq_action_mode = self.zmq_config.get('action_mode', 'dof')
        self.zmq_motion_axis_2dof = self.zmq_config.get('motion_axis_2dof', 'sb')
        self.zmq_warmup_time = float(self.zmq_config.get('warmup_time', 0.0) or 0.0)
        self.safety_shield_config = self.zmq_config.get('safety_shield', {})
        self.max_angle = PHYSICS_CONFIG['max_angle']
        self.max_displacement = PHYSICS_CONFIG['max_displacement']
        self.cfd_time_offset = 0.0
        self.max_steps = int(PHYSICS_CONFIG['simulation_time'] / PHYSICS_CONFIG['action_interval'])
        self.current_step = 0
        self.simulation_time = 0.0
        self.save_dir = save_dir
        self.current_episode = 0
        self.last_total_cl = 0.0
        self.current_zmq_motion = None

        # 为气动数据创建专门的保存目录
        if self.save_dir:
            self.aero_save_dir = os.path.join(self.save_dir, 'aerodynamics_data')
            os.makedirs(self.aero_save_dir, exist_ok=True)
        else:
            self.aero_save_dir = None

        # 根据自由度模式和翼数量确定状态和动作维度。观测为“当前状态 + 上一步状态”。
        if dof_mode == '1dof':
            self.current_state_dim = 3 * self.wing_count
            self.action_dim = self.wing_count
        elif dof_mode == '2dof':
            self.current_state_dim = 6 * self.wing_count
            self.action_dim = 2 * self.wing_count
        else:  # 3dof
            self.current_state_dim = 9 * self.wing_count
            self.action_dim = 3 * self.wing_count
        self.state_dim = 2 * self.current_state_dim

        if self.zmq_action_mode == 'all':
            self.action_dim = self.wing_count * 3

        # 初始化在线 CFD 模型
        self.aerodynamics = AerodynamicsModel(save_dir=self.aero_save_dir)
        self.reward_function = RewardFunction()
        self.previous_cfd_current_state = np.zeros(self.current_state_dim)

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
        """使用在线 CFD 返回的运动/气动力推进一个控制周期。"""
        self.current_step += 1
        action = np.array(action, dtype=float).reshape(-1)
        target_time = self._target_time_for_step(self.current_step)
        raw_control_values = self._action_to_zmq_controls(action)
        control_values, safety_shield_active = self._apply_zmq_safety_shield(raw_control_values)

        result = self.aerodynamics.step_zmq(
            target_time=target_time,
            control_values=control_values,
            episode=self.current_episode,
            rl_step=self.current_step,
        )
        response = result["response"]
        current_state = self._state_from_zmq_response(response)
        self.simulation_time = response["time"]

        cls = result["cls"]
        cts = result["cts"]
        cps = result["cps"]
        total_ct = result["total_ct"]
        total_cp = float(sum(cps))
        forces = result["forces"]

        stab = 0.0
        if self.current_step > 1:
            total_cl = float(sum(cls))
            stab = (total_cl - self.last_total_cl) ** 2
        self.last_total_cl = float(sum(cls))

        motion_params = self._get_motion_params(current_state)
        reached_boundary = not self._is_zmq_motion_valid(response["motion_last"])
        reward = -500.0 if reached_boundary else self.reward_function.calculate_reward(
            total_ct, motion_params, action)

        if self.visualizer is not None:
            self.visualizer.update(
                current_state, None, reward, self.current_step, action,
                wing_cts=cts,
                wing_cps=cps,
                wing_cls=cls,
                total_ct=total_ct,
                stab=stab,
            )

        done = reached_boundary or self.current_step >= self.max_steps
        termination_reason = None
        if done:
            if reached_boundary:
                termination_reason = self._get_termination_reason(current_state)
            else:
                termination_reason = "达到最大步数"

        info = {
            'total_ct': total_ct,
            'wing_count': self.wing_count,
            'wings': [
                {
                    'ct': cts[idx],
                    'cl': cls[idx],
                    'cp': cps[idx],
                    'forces': forces[idx],
                }
                for idx in range(self.wing_count)
            ],
            'wing_cts': cts,
            'wing_cls': cls,
            'wing_cps': cps,
            'wing_forces': forces,
            'stable': stab,
            'total_power_coefficient': total_cp,
            'is_valid': not reached_boundary,
            'simulation_time': self.simulation_time,
            'termination_reason': termination_reason,
            'boundary_violation': reached_boundary,
            'reward': reward,
            'cfd_response': response,
            'flow_save_response': result.get("save_response"),
            'zmq_control_values': control_values,
            'zmq_control_values_raw': raw_control_values,
            'safety_shield_active': safety_shield_active,
        }
        return current_state, reward, done, info

    def _action_to_zmq_controls(self, action):
        scaled = np.clip(action, -1.0, 1.0) * PHYSICS_CONFIG['max_velocity_increment']
        if self.zmq_action_mode == 'all':
            expected = self.wing_count * 3
            if len(scaled) != expected:
                raise ValueError(f"ZMQ all-DOF action length must be {expected}, got {len(scaled)}")
            return scaled.tolist()

        controls = np.zeros(self.wing_count * 3, dtype=float)
        if self.dof_mode == '1dof':
            expected = self.wing_count
            if len(scaled) != expected:
                raise ValueError(f"1dof action length must be {expected}, got {len(scaled)}")
            for wing_idx in range(self.wing_count):
                controls[3 * wing_idx] = scaled[wing_idx]
        elif self.dof_mode == '2dof':
            expected = 2 * self.wing_count
            if len(scaled) != expected:
                raise ValueError(f"2dof action length must be {expected}, got {len(scaled)}")
            for wing_idx in range(self.wing_count):
                action_base = 2 * wing_idx
                control_base = 3 * wing_idx
                controls[control_base] = scaled[action_base + 1]
                if self.zmq_motion_axis_2dof == 'sa':
                    controls[control_base + 1] = scaled[action_base]
                else:
                    controls[control_base + 2] = scaled[action_base]
        else:
            expected = 3 * self.wing_count
            if len(scaled) != expected:
                raise ValueError(f"3dof action length must be {expected}, got {len(scaled)}")
            controls[:] = scaled
        return controls.tolist()

    def _apply_zmq_safety_shield(self, control_values):
        """裁剪会让下一步预测位置越过安全边界的速度增量。"""
        if not self.safety_shield_config.get('enabled', True):
            return list(control_values), False
        if self.current_zmq_motion is None:
            return list(control_values), False

        dt = PHYSICS_CONFIG['action_interval']
        max_increment = PHYSICS_CONFIG['max_velocity_increment']
        guard_ratio = float(self.safety_shield_config.get('guard_ratio', 0.95))
        guard_ratio = float(np.clip(guard_ratio, 0.0, 1.0))

        controls = np.array(control_values, dtype=float).reshape(-1)
        original_controls = controls.copy()
        axes = [
            ('angle', 'omega', self.max_angle * guard_ratio, 0),
            ('sa', 'sau', self.max_displacement * guard_ratio, 1),
            ('sb', 'sbv', self.max_displacement * guard_ratio, 2),
        ]

        for wing_idx, wing_motion in enumerate(self.current_zmq_motion):
            control_base = 3 * wing_idx
            for pos_key, vel_key, limit, control_offset in axes:
                control_idx = control_base + control_offset
                pos = float(wing_motion[pos_key])
                vel = float(wing_motion[vel_key])

                min_delta = (-limit - pos) / dt - vel
                max_delta = (limit - pos) / dt - vel

                lower = max(-max_increment, min_delta)
                upper = min(max_increment, max_delta)
                if lower <= upper:
                    controls[control_idx] = np.clip(controls[control_idx], lower, upper)
                elif max_delta < -max_increment:
                    controls[control_idx] = -max_increment
                elif min_delta > max_increment:
                    controls[control_idx] = max_increment

        safety_shield_active = bool(np.any(np.abs(controls - original_controls) > 1e-12))
        return controls.tolist(), safety_shield_active

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

        zero_controls = [0.0] * (self.wing_count * 3)
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
        self.current_zmq_motion = [dict(wing_motion) for wing_motion in motion]
        current = self._current_state_from_zmq_motion(motion)
        state = np.concatenate([current, self.previous_cfd_current_state])
        self.previous_cfd_current_state = current.copy()
        return state.astype(np.float32)

    def _current_state_from_zmq_motion(self, motion):
        if self.dof_mode == '1dof':
            values = []
            values.extend(m['angle'] for m in motion)
            values.extend(m['omega'] for m in motion)
            values.extend(m['omgac'] for m in motion)
            return np.array(values, dtype=float)

        if self.dof_mode == '2dof':
            pos_key = self.zmq_motion_axis_2dof
            vel_key = 'sau' if pos_key == 'sa' else 'sbv'
            acc_key = 'saac' if pos_key == 'sa' else 'sbac'
            values = []
            for m in motion:
                values.extend([m[pos_key], m['angle']])
            for m in motion:
                values.extend([m[vel_key], m['omega']])
            for m in motion:
                values.extend([m[acc_key], m['omgac']])
            return np.array(values, dtype=float)

        values = []
        for m in motion:
            values.extend([m['sa'], m['sb'], m['angle']])
        for m in motion:
            values.extend([m['sau'], m['sbv'], m['omega']])
        for m in motion:
            values.extend([m['saac'], m['sbac'], m['omgac']])
        return np.array(values, dtype=float)

    def _is_zmq_motion_valid(self, motion):
        for wing_motion in motion:
            if abs(wing_motion['angle']) > self.max_angle:
                return False
            if abs(wing_motion['sa']) > self.max_displacement:
                return False
            if abs(wing_motion['sb']) > self.max_displacement:
                return False
        return True

    def _get_motion_params(self, state):
        """构建运动参数字典"""
        motion_params = {}
        if self.dof_mode == '1dof':
            for wing_idx in range(self.wing_count):
                motion_params[f'wing{wing_idx + 1}'] = {
                    'pitch': state[wing_idx],
                    'heave': 0,
                    'inline': 0,
                }
        elif self.dof_mode == '2dof':
            for wing_idx in range(self.wing_count):
                base = 2 * wing_idx
                motion_params[f'wing{wing_idx + 1}'] = {
                    'pitch': state[base + 1],
                    'heave': state[base],
                    'inline': 0,
                }
        else:  # 3dof
            for wing_idx in range(self.wing_count):
                base = 3 * wing_idx
                motion_params[f'wing{wing_idx + 1}'] = {
                    'pitch': state[base + 2],
                    'heave': state[base + 1],
                    'inline': state[base],
                }
        return motion_params

    def _get_termination_reason(self, state):
        """获取终止原因的详细信息"""
        if self.dof_mode == '1dof':
            angles = [abs(state[i]) for i in range(self.wing_count)]
            if max(angles) > self.max_angle:
                return f"达到运动边界：最大俯仰角 = {max(angles):.2f} rad (最大值：{self.max_angle:.2f} rad)"
        elif self.dof_mode == '2dof':
            heaves = [abs(state[2 * i]) for i in range(self.wing_count)]
            angles = [abs(state[2 * i + 1]) for i in range(self.wing_count)]
            if max(angles) > self.max_angle:
                return f"达到运动边界：最大俯仰角 = {max(angles):.2f} rad (最大值：{self.max_angle:.2f} rad)"
            return f"达到运动边界：最大升沉 = {max(heaves):.2f} m (最大值：{self.max_displacement:.2f} m)"
        else:  # 3dof
            inlines = [abs(state[3 * i]) for i in range(self.wing_count)]
            heaves = [abs(state[3 * i + 1]) for i in range(self.wing_count)]
            angles = [abs(state[3 * i + 2]) for i in range(self.wing_count)]
            if max(angles) > self.max_angle:
                return f"达到运动边界：最大俯仰角 = {max(angles):.2f} rad (最大值：{self.max_angle:.2f} rad)"
            if max(heaves) > self.max_displacement:
                return f"达到运动边界：最大升沉 = {max(heaves):.2f} m (最大值：{self.max_displacement:.2f} m)"
            return f"达到运动边界：最大轴向位移 = {max(inlines):.2f} m (最大值：{self.max_displacement:.2f} m)"

    def reset(self):
        """重置环境到初始状态"""
        # 保存当前回合的数据
        self.aerodynamics.reset()  # 这会触发数据保存

        # 重置状态
        self.current_step = 0
        self.simulation_time = 0.0
        self.cfd_time_offset = 0.0
        self.last_total_cl = 0.0
        self.previous_cfd_current_state = np.zeros(self.current_state_dim)
        response = self.aerodynamics.reset_zmq()
        return self._run_zmq_warmup(response)

    def set_episode(self, episode):
        """设置当前回合号"""
        self.current_episode = episode
        self.aerodynamics.set_episode(episode)

    def render(self, mode='human'):
        """渲染环境（由外部可视化器处理）"""
        pass

    def close(self):
        """关闭环境"""
        self.aerodynamics.close(stop_solver=True)
