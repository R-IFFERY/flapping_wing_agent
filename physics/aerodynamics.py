"""
扑翼系统的空气动力学模型。
提供气动力和推力系数的计算。
"""
import numpy as np
import pandas as pd
import os
from config.config import PHYSICS_CONFIG, ENV_CONFIG
from physics.lbm_client import LBMClient

class AerodynamicsModel:
    def __init__(self, use_cfd=False, save_dir=None):
        self.use_cfd = use_cfd
        self.rou = PHYSICS_CONFIG['fluid_density']  # 流体密度
        self.c = PHYSICS_CONFIG['chord_length']    # 弦长
        self.U0 = PHYSICS_CONFIG['fluid_velocity'] # 来流速度

        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"气动数据将保存到: {self.save_dir}")

        self.step_counter = 0
        self.data_buffer = []
        self.current_episode = 0
        self.total_steps = int(PHYSICS_CONFIG['simulation_time'] / PHYSICS_CONFIG['dt'])
        self.client = None
        self.last_force_for_save = None
        self.last_zmq_response = None
        self.last_save_response = None
        self.pending_flow_save_reason = None
        self.pending_flow_save_retry_count = 0
        self.zmq_config = ENV_CONFIG.get('cfd_zmq', {})
        self.flow_save_config = self.zmq_config.get('flow_save', {})

        if self.is_zmq:
            self.client = LBMClient(
                host=self.zmq_config.get('host', '127.0.0.1'),
                port=self.zmq_config.get('port', 5555),
                wing_count=self.zmq_config.get('wing_count', 2),
                timeout_ms=self.zmq_config.get('timeout_ms', 600000),
            )

    @property
    def is_zmq(self):
        return self.use_cfd

    def set_episode(self, episode):
        """设置当前回合号，同时保存上一回合数据"""
        # 如果有未保存的数据，先保存
        if self.data_buffer and self.save_dir:
            self._save_data()

        # 设置新回合
        self.current_episode = episode
        print(f"开始记录 {episode} 回合的气动数据")

        # 重置步数计数器和数据缓存
        self.step_counter = 0
        self.data_buffer = []
        self.last_force_for_save = None
        self.pending_flow_save_reason = None
        self.pending_flow_save_retry_count = 0

    def _record_data(self, data_dict):
        """记录计算数据"""
        if not self.save_dir:  # 如果没有设置保存路径，跳过记录
            return

        data_dict['step'] = self.step_counter
        data_dict['episode'] = self.current_episode
        data_dict['time'] = self.step_counter * PHYSICS_CONFIG['dt']  # 添加时间信息
        self.data_buffer.append(data_dict)
        self.step_counter += 1

    def _save_data(self):
        """保存数据到CSV文件"""
        if not self.save_dir or not self.data_buffer:  # 如果没有保存路径或没有数据，直接返回
            return

        try:
            # 使用pandas DataFrame保存所有数据
            df = pd.DataFrame(self.data_buffer)

            # 保存到对应回合的文件中
            filename = os.path.join(
                self.save_dir,
                f'episode_{self.current_episode}_aero_data.csv'
            )
            df.to_csv(filename, index=False)
            print(f"成功保存第 {self.current_episode} 回合的气动数据到: {filename}")

            # 清空缓存
            self.data_buffer = []
        except Exception as e:
            print(f"保存气动数据时发生错误: {e}")

    def calculate_forces(self, state1, state2, sim_time):
        """计算气动力和推力系数"""
        if self.use_cfd:
            raise RuntimeError("ZMQ CFD mode must call step_zmq(action, target_time, ...)")
        return self._quasi_steady_solve(state1, state2, sim_time)

    def reset_zmq(self):
        """Reset the online CFD solver and return the initial FORCE response."""
        if not self.client:
            raise RuntimeError("ZMQ client is not initialized")
        response = self.client.reset()
        if response.get("type") == "ERROR":
            raise RuntimeError(f"CFD RESET failed: {response.get('message')}")
        self.last_zmq_response = response
        self.last_save_response = None
        self.last_force_for_save = None
        self.pending_flow_save_reason = None
        self.pending_flow_save_retry_count = 0
        return response

    def step_zmq(self, target_time, control_values, episode=None, rl_step=None,
                 record=True, allow_flow_save=True):
        """Advance the CFD solver by one control period using velocity increments."""
        if not self.client:
            raise RuntimeError("ZMQ client is not initialized")

        response = self.client.step(target_time, control_values)
        self.last_zmq_response = response
        self.last_save_response = None

        if response.get("type") == "ERROR":
            raise RuntimeError(f"CFD STEP failed at t={target_time}: {response.get('message')}")
        if response.get("type") != "FORCE":
            raise RuntimeError(f"unexpected CFD STEP response: {response}")

        result = self._coefficients_from_zmq_force(response, control_values)
        save_reason = self._flow_save_reason(episode, rl_step, response) if allow_flow_save else None
        if save_reason:
            save_result = self._try_save_flow(save_reason)
            result["data"].update(save_result["data"])
            result["save_response"] = save_result["response"]

        if record:
            self._record_data(result["data"])
        return result

    def _coefficients_from_zmq_force(self, response, control_values):
        """Convert a FORCE response to the legacy force/coefficient tuple."""
        wing_force = response["wing_force"]
        total_force = response["total_force"]
        motion = response["motion_last"]
        q = 0.5 * self.rou * self.U0 ** 2 * self.c
        p_den = 0.5 * self.rou * self.c * self.U0 ** 3

        fx1 = wing_force[0]["fx_avg"]
        fy1 = wing_force[0]["fy_avg"]
        torque1 = wing_force[0]["m_avg"]
        fx2 = wing_force[1]["fx_avg"]
        fy2 = wing_force[1]["fy_avg"]
        torque2 = wing_force[1]["m_avg"]

        if q == 0:
            Ct1 = Ct2 = Cl1 = Cl2 = 0.0
        else:
            Ct1 = fx1 / q
            Ct2 = fx2 / q
            Cl1 = fy1 / q
            Cl2 = fy2 / q

        p1 = fx1 * motion[0]["sau"] + fy1 * motion[0]["sbv"] + torque1 * motion[0]["omega"]
        p2 = fx2 * motion[1]["sau"] + fy2 * motion[1]["sbv"] + torque2 * motion[1]["omega"]
        if p_den == 0:
            Cp1 = Cp2 = 0.0
        else:
            Cp1 = max(min(p1 / p_den, 20.0), -20.0)
            Cp2 = max(min(p2 / p_den, 20.0), -20.0)

        cp_sum = Cp1 + Cp2
        Pf1 = 0.0 if Cp1 == 0 else Ct1 / Cp1
        Pf2 = 0.0 if Cp2 == 0 else Ct2 / Cp2
        Pf = 0.0 if cp_sum == 0 else max(min((Ct1 + Ct2) / cp_sum, 20.0), -20.0)

        data = {
            'cfd_time': response['time'],
            'theta1': motion[0]['angle'],
            'theta2': motion[1]['angle'],
            'omega1': motion[0]['omega'],
            'omega2': motion[1]['omega'],
            'alpha1': motion[0]['omgac'],
            'alpha2': motion[1]['omgac'],
            'sa1': motion[0]['sa'],
            'sa2': motion[1]['sa'],
            'sau1': motion[0]['sau'],
            'sau2': motion[1]['sau'],
            'saac1': motion[0]['saac'],
            'saac2': motion[1]['saac'],
            'sb1': motion[0]['sb'],
            'sb2': motion[1]['sb'],
            'sbv1': motion[0]['sbv'],
            'sbv2': motion[1]['sbv'],
            'sbac1': motion[0]['sbac'],
            'sbac2': motion[1]['sbac'],
            'v_x1': motion[0]['sau'],
            'v_x2': motion[1]['sau'],
            'v_y1': motion[0]['sbv'],
            'v_y2': motion[1]['sbv'],
            'a_x1': motion[0]['saac'],
            'a_x2': motion[1]['saac'],
            'a_y1': motion[0]['sbac'],
            'a_y2': motion[1]['sbac'],
            'Fx1': fx1,
            'Fx2': fx2,
            'Fy1': fy1,
            'Fy2': fy2,
            'M1': torque1,
            'M2': torque2,
            'total_Fx': total_force['fx_avg'],
            'total_Fy': total_force['fy_avg'],
            'total_M': total_force['m_avg'],
            'T1': fx1,
            'T2': fx2,
            'Ct1': Ct1,
            'Ct2': Ct2,
            'P1': p1,
            'P2': p2,
            'Cp1': Cp1,
            'Cp2': Cp2,
            'Pf1': Pf1,
            'Pf2': Pf2,
            'Cl1': Cl1,
            'Cl2': Cl2,
            'Pf': Pf,
            'control_w1_domega': control_values[0],
            'control_w1_dsau': control_values[1],
            'control_w1_dsbv': control_values[2],
            'control_w2_domega': control_values[3],
            'control_w2_dsau': control_values[4],
            'control_w2_dsbv': control_values[5],
            'density': self.rou,
            'chord_length': self.c,
            'freestream_velocity': self.U0,
            'source': 'cfd_zmq',
            'flow_save_requested': False,
            'flow_save': False,
            'flow_save_message': '',
            'flow_save_reason': '',
            'flow_save_attempts': 0,
        }

        return {
            "forces1": np.array([fx1, fy1]),
            "cl1": Cl1,
            "ct1": Ct1,
            "cp1": Cp1,
            "forces2": np.array([fx2, fy2]),
            "cl2": Cl2,
            "ct2": Ct2,
            "cp2": Cp2,
            "pf": Pf,
            "data": data,
            "response": response,
        }

    def _flow_save_reason(self, episode, rl_step, force_response):
        cfg = self.flow_save_config
        if not cfg.get('enabled', False):
            self._update_last_force_for_save(force_response)
            return None

        if self.pending_flow_save_reason is not None:
            self._update_last_force_for_save(force_response)
            return f"retry:{self.pending_flow_save_reason}"

        reasons = []
        if self._flow_schedule_match(cfg, episode, rl_step):
            reasons.append("schedule")
        save_time = self._flow_time_match(cfg, episode, rl_step)
        if save_time is not None:
            reasons.append(f"time:{save_time:g}")
        force_delta = self._flow_force_delta(cfg, force_response)
        if force_delta is not None:
            reasons.append(f"force_delta:{force_delta:.6g}")
        self._update_last_force_for_save(force_response)
        return ",".join(reasons) if reasons else None

    def _flow_schedule_match(self, cfg, episode, rl_step):
        episodes = cfg.get('episodes') or []
        steps = cfg.get('steps') or []
        episode_interval = int(cfg.get('episode_interval', 1) or 1)
        step_interval = int(cfg.get('step_interval', 0) or 0)

        episode_ok = True
        if episode is not None:
            episode_ok = (not episodes and episode % episode_interval == 0) or episode in episodes

        step_ok = False
        if rl_step is not None:
            step_ok = (step_interval > 0 and rl_step % step_interval == 0) or rl_step in steps

        return episode_ok and step_ok

    def _flow_time_match(self, cfg, episode, rl_step):
        times = cfg.get('times') or []
        if rl_step is None or not times:
            return None

        episodes = cfg.get('episodes') or []
        episode_interval = int(cfg.get('episode_interval', 1) or 1)
        if episode is not None:
            episode_ok = (not episodes and episode % episode_interval == 0) or episode in episodes
            if not episode_ok:
                return None

        formal_time = rl_step * PHYSICS_CONFIG['action_interval']
        tolerance = max(1e-9, PHYSICS_CONFIG['action_interval'] * 1e-6)
        for save_time in times:
            if abs(formal_time - float(save_time)) <= tolerance:
                return float(save_time)
        return None

    def _flow_force_delta(self, cfg, force_response):
        threshold = cfg.get('force_delta_threshold')
        if threshold is None or self.last_force_for_save is None:
            return None

        current = self._force_vector(force_response)
        delta = np.linalg.norm(current - self.last_force_for_save)
        return delta if delta >= float(threshold) else None

    def _try_save_flow(self, reason):
        attempt_no = self.pending_flow_save_retry_count + 1
        self.last_save_response = self.client.save()
        save_type = self.last_save_response.get("type")
        message = self.last_save_response.get("message", "")
        success = save_type == "STATUS" and message.startswith("FLOW_SAVED")

        if success:
            self.pending_flow_save_reason = None
            self.pending_flow_save_retry_count = 0
        else:
            self._mark_pending_flow_save(reason)

        return {
            "response": self.last_save_response,
            "data": {
                "flow_save_requested": True,
                "flow_save": success,
                "flow_save_message": message,
                "flow_save_reason": reason,
                "flow_save_attempts": attempt_no,
            }
        }

    def _mark_pending_flow_save(self, reason):
        max_retries = int(self.flow_save_config.get('max_retries', 3) or 0)
        if reason.startswith("retry:"):
            reason = reason[len("retry:"):]
        self.pending_flow_save_retry_count += 1
        if max_retries > 0 and self.pending_flow_save_retry_count >= max_retries:
            self.pending_flow_save_reason = None
            self.pending_flow_save_retry_count = 0
            return
        self.pending_flow_save_reason = reason

    @staticmethod
    def _force_vector(force_response):
        total = force_response["total_force"]
        return np.array([total["fx_avg"], total["fy_avg"], total["m_avg"]], dtype=float)

    def _update_last_force_for_save(self, force_response):
        self.last_force_for_save = self._force_vector(force_response)

    def _quasi_steady_solve(self, state1, state2, sim_time):
        """准定常解算器，返回与环境层兼容的数据结构和返回值格式。

        接受两个机翼的状态（state1, state2），在 1/2/3DOF 模式下兼容原有状态格式。
        返回：(np.array([Fx1, Fy1]), Cl1, Ct1, Cp1, np.array([Fx2, Fy2]), Cl2, Ct2, Cp2, Pf)
        """
        # 内部帮助函数：对单翼状态计算力学量（重用原来准定常的计算逻辑）
        def compute_single(state):
            # 初始化
            theta = omega = alpha = 0.0
            v_x = self.U0
            v_y = 0.0
            a_x = a_y = 0.0

            # 根据自由度提取
            if len(state) == 6:  # 1DOF
                theta = state[0]
                omega = state[1]
                alpha = state[2]
                v_x = self.U0
                v_y = 0.0
            elif len(state) == 12:  # 2DOF
                # [y, theta, vy, omega, ay, alpha, prev_...]
                theta = state[1]
                v_y = state[2]
                omega = state[3]
                a_y = state[4]
                alpha = state[5]
                v_x = self.U0
            else:  # 3DOF
                theta = state[2]
                v_x = state[3] + self.U0
                v_y = state[4]
                omega = state[5]
                a_x = state[6]
                a_y = state[7]
                alpha = state[8]

            # 速度分量在机翼坐标系的转换
            u_x = v_x * np.cos(theta) + v_y * np.sin(theta)
            u_y = -v_x * np.sin(theta) + v_y * np.cos(theta)
            u_bar = np.sqrt(u_x ** 2 + u_y ** 2)
            AOAe = np.arctan2(u_y, u_x)

            # 几何/经验气动系数（复用原公式）
            CL = 1.7 * np.sin(2 * AOAe)
            CD = 0.5 * (3.24 + 0.05) - 0.5 * (3.24 - 0.05) * np.cos(2 * AOAe)
            CH = np.cos(AOAe) * CD - np.sin(AOAe) * CL
            CV = np.sin(AOAe) * CD + np.cos(AOAe) * CL

            lamty = 0.25 * np.pi * self.rou * self.c ** 2
            lamtyo = 0.25 * np.pi * self.rou * (-0.2) * self.c ** 3
            Crot = 1.0

            dfxt_r = -0.5 * self.rou * u_bar ** 2 * CH * self.c
            dfyt_r = 0.5 * self.rou * u_bar ** 2 * CV * self.c
            dfxa_r = -(lamty * u_y * omega - lamtyo * omega ** 2)
            dfya_r = (-lamty * a_y + lamtyo * alpha)
            dfyr_r = Crot * self.rou * u_bar * omega * self.c ** 2

            Fx = dfxt_r + dfxa_r
            Fy = dfyt_r + dfya_r + dfyr_r

            # 推力（局部转到全局）
            T = Fx * np.cos(theta) + Fy * np.sin(theta)

            # 推力系数
            q = 0.5 * self.rou * v_x ** 2
            if q == 0:
                Ct = 0.0
            else:
                Ct = T / (q * self.c)
            Ct = np.clip(Ct, -PHYSICS_CONFIG['C_t_set'], PHYSICS_CONFIG['C_t_set'])

            # 升力系数 CL（无量纲）沿用上面 CL
            Cl = CL

            # 气动功率（假设转矩项为0）
            torque = 0.0
            P = Fx * v_x + Fy * v_y + torque * omega

            # 功率系数 Cp
            if v_x == 0:
                Cp = 0.0
            else:
                Cp = P / (0.5 * self.rou * self.c * v_x ** 3)
                Cp = max(min(Cp, 20.0), -20.0)

            # 功率效率 Pf（Ct/Cp）
            if Cp == 0:
                Pf_local = 0.0
            else:
                Pf_local = Ct / Cp

            # 返回所有量与中间量（用于组装记录结构）
            return {
                'theta': theta,
                'omega': omega,
                'alpha': alpha,
                'v_x': v_x,
                'v_y': v_y,
                'a_x': a_x,
                'a_y': a_y,

                'u_x': u_x,
                'u_y': u_y,
                'u_bar': u_bar,
                'AOAe': AOAe,
                'CL': CL,
                'CD': CD,
                'CH': CH,
                'CV': CV,
                'lamty': lamty,
                'lamtyo': lamtyo,
                'dfxt_r': dfxt_r,
                'dfyt_r': dfyt_r,
                'dfxa_r': dfxa_r,
                'dfya_r': dfya_r,
                'dfyr_r': dfyr_r,

                'Fx': Fx,
                'Fy': Fy,
                'T': T,
                'q': q,
                'Ct': Ct,
                'Cl': Cl,
                'P': P,
                'Cp': Cp,
                'Pf': Pf_local
            }

        # 计算两翼
        res1 = compute_single(state1)
        res2 = compute_single(state2)

        # 组装与 CFD 相同的数据字典字段名
        data = {
            'theta1': res1['theta'],
            'theta2': res2['theta'],
            'omega1': res1['omega'],
            'omega2': res2['omega'],
            'alpha1': res1['alpha'],
            'alpha2': res2['alpha'],
            'v_x1': res1['v_x'],
            'v_x2': res2['v_x'],
            'v_y1': res1['v_y'],
            'v_y2': res2['v_y'],
            'a_x1': res1['a_x'],
            'a_x2': res2['a_x'],
            'a_y1': res1['a_y'],
            'a_y2': res2['a_y'],
            'Fx1': res1['Fx'],
            'Fx2': res2['Fx'],
            'Fy1': res1['Fy'],
            'Fy2': res2['Fy'],
            'T1': res1['T'],
            'T2': res2['T'],
            'Ct1': res1['Ct'],
            'Ct2': res2['Ct'],
            'P1': res1['P'],
            'P2': res2['P'],
            'Cp1': res1['Cp'],
            'Cp2': res2['Cp'],
            'Pf1': res1['Pf'],
            'Pf2': res2['Pf'],
            'Cl1': res1['Cl'],
            'Cl2': res2['Cl'],
            'Pf': (res1['Pf'] + res2['Pf']) / 2.0 if (res1['Pf'] is not None and res2['Pf'] is not None) else 0.0,
            'density': self.rou,
            'chord_length': self.c,
            'freestream_velocity': self.U0,
            'source': 'quasi'
        }

        # 记录并返回与 CFD 相同的返回结构
        self._record_data(data)

        return np.array([res1['Fx'], res1['Fy']]), res1['Cl'], res1['Ct'], res1['Cp'], np.array([res2['Fx'], res2['Fy']]), res2['Cl'], res2['Ct'], res2['Cp'], data['Pf']

    def reset(self):
        """重置模型状态并保存数据"""
        if self.save_dir and self.data_buffer:
            self._save_data()
        self.step_counter = 0
        self.last_force_for_save = None

    def close(self, stop_solver=False):
        """Close resources owned by the aerodynamic model."""
        if self.save_dir and self.data_buffer:
            self._save_data()
        if self.client:
            if stop_solver:
                try:
                    self.client.stop()
                except Exception as e:
                    print(f"发送 CFD STOP 时发生错误: {e}")
            self.client.close()
            self.client = None
