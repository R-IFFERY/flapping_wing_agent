"""
扑翼系统的奖励函数模块。
实现基于推力系数和运动约束的奖励计算。
"""
import numpy as np
from typing import Dict
from config.config import PHYSICS_CONFIG

class RewardFunction:
    def __init__(self):
        # 从物理配置中加载参数
        self.C_t_set = 2*PHYSICS_CONFIG['C_t_set']  # 基线推力系数
        self.max_pitch = PHYSICS_CONFIG['max_angle']  # 最大俯仰角（弧度）
        self.max_heave = PHYSICS_CONFIG['max_displacement']  # 最大升沉位移
        self.max_inline = PHYSICS_CONFIG['max_displacement']  # 最大轴向位移

        # 惩罚阈值
        self.pitch_penalty_threshold = self.max_pitch * 2/3
        self.heave_penalty_threshold = self.max_heave * 0.7
        self.inline_penalty_threshold = self.max_inline * 0.7

        # 惩罚权重
        self.beta_alpha = PHYSICS_CONFIG['pitch_penalty_weight']  # 俯仰角惩罚权重
        self.beta_y = PHYSICS_CONFIG['heave_penalty_weight']     # 升沉位移惩罚权重
        self.beta_x = PHYSICS_CONFIG['inline_penalty_weight']    # 轴向位移惩罚权重

    def calculate_reward(self, total_ct: float, motion_params: Dict[str, dict],
                        action: np.ndarray) -> float:
        """
        计算当前状态和动作的奖励

        参数:
            total_ct: 当前总推力系数 C_t（全部翼之和）
            motion_params: 运动参数字典，包含每个翼各自的参数
            action: 执行的动作

        返回:
            reward: 总奖励值
        """
        # # 1. 推力奖励：最大化总推力系数，但不超过基准值
        # if total_ct < -self.C_t_set:
        #     thrust_reward = -self.C_t_set
        # elif total_ct > self.C_t_set:
        #     thrust_reward = self.C_t_set
        # else:
        #     thrust_reward = total_ct

        # 1. 推力奖励：最大化总推力系数
        thrust_reward = total_ct

        # 2. 稳定性惩罚


        # 2. 运动惩罚（分别计算前后翼的惩罚）
        penalty = 0.0

        # 对每个翼单独计算惩罚
        for wing_params in motion_params.values():
            # 2.1 俯仰角惩罚
            if 'pitch' in wing_params:
                pitch_angle = abs(wing_params['pitch'])
                if pitch_angle >= self.max_pitch:
                    penalty -= 55  # 超出最大角度的强惩罚
                elif pitch_angle > self.pitch_penalty_threshold:
                    penalty -= 3 * pitch_angle ** 7

            # 2.2 升沉位移惩罚
            if 'heave' in wing_params:
                heave_pos = abs(wing_params['heave'])
                if heave_pos >= self.max_heave:
                    penalty -= 55  # 超出最大位移的强惩罚
                elif heave_pos > self.heave_penalty_threshold:
                    penalty -= 0.01 * heave_pos ** 9

            # 2.3 轴向位移惩罚
            if 'inline' in wing_params:
                inline_pos = abs(wing_params['inline'])
                if inline_pos >= self.max_inline:
                    penalty -= 55  # 超出最大位移的强惩罚
                elif inline_pos > self.inline_penalty_threshold:
                    penalty -= 0.01 * inline_pos ** 9

        # 3. 计算总奖励
        total_reward = thrust_reward + penalty

        return float(total_reward)

    def get_reward_info(self, total_ct: float, motion_params: Dict[str, dict],
                       action: np.ndarray) -> Dict[str, float]:
        """
        获取奖励计算的详细信息

        参数:
            total_ct: 当前总推力系数 C_t
            motion_params: 运动参数字典，前后翼各自的参数
            action: 执行的动作

        返回:
            reward_info: 包含各个奖励分量的字典
        """
        info = {}

        # 1. 推力奖励
        if total_ct < -self.C_t_set:
            thrust_reward = -self.C_t_set
        elif total_ct > self.C_t_set:
            thrust_reward = self.C_t_set
        else:
            thrust_reward = total_ct
        info['thrust_reward'] = float(thrust_reward)

        # 2. 运动惩罚
        for wing_id, wing_params in motion_params.items():
            wing_penalty = 0.0

            # 2.1 俯仰角惩罚
            if 'pitch' in wing_params:
                pitch_angle = abs(wing_params['pitch'])
                if pitch_angle >= self.max_pitch:
                    wing_penalty -= 1000
                elif pitch_angle > self.pitch_penalty_threshold:
                    wing_penalty -= 0.4 * pitch_angle ** 7
                info[f'{wing_id}_pitch_penalty'] = float(wing_penalty)

            # 2.2 升沉位移惩罚
            if 'heave' in wing_params:
                heave_pos = abs(wing_params['heave'])
                if heave_pos >= self.max_heave:
                    wing_penalty -= 1000
                elif heave_pos > self.heave_penalty_threshold:
                    wing_penalty -= 0.01 * heave_pos ** 9
                info[f'{wing_id}_heave_penalty'] = float(wing_penalty)

            # 2.3 轴向位移惩罚
            if 'inline' in wing_params:
                inline_pos = abs(wing_params['inline'])
                if inline_pos >= self.max_inline:
                    wing_penalty -= 1000
                elif inline_pos > self.inline_penalty_threshold:
                    wing_penalty -= 0.01 * inline_pos ** 9
                info['inline_penalty'] = float(wing_penalty)

        # 3. 总奖励
        info['total_reward'] = float(thrust_reward + sum(value for key, value in info.items() if 'penalty' in key))

        return info
