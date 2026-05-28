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
    def __init__(self, save_dir=None):
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
        self.wing_count = int(self.zmq_config.get('wing_count', 2))
        self.flow_save_config = self.zmq_config.get('flow_save', {})

        self.client = LBMClient(
            host=self.zmq_config.get('host', '127.0.0.1'),
            port=self.zmq_config.get('port', 5555),
            wing_count=self.wing_count,
            timeout_ms=self.zmq_config.get('timeout_ms', 600000),
        )

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

        fxs = []
        fys = []
        torques = []
        cts = []
        cls = []
        cps = []
        powers = []
        pfs = []

        for idx in range(self.wing_count):
            fx = wing_force[idx]["fx_avg"]
            fy = wing_force[idx]["fy_avg"]
            torque = wing_force[idx]["m_avg"]
            fxs.append(fx)
            fys.append(fy)
            torques.append(torque)

            if q == 0:
                ct = cl = 0.0
            else:
                ct = fx / q
                cl = fy / q
            cts.append(ct)
            cls.append(cl)

            power = fx * motion[idx]["sau"] + fy * motion[idx]["sbv"] + torque * motion[idx]["omega"]
            powers.append(power)
            cp = 0.0 if p_den == 0 else max(min(power / p_den, 20.0), -20.0)
            cps.append(cp)
            pfs.append(0.0 if cp == 0 else ct / cp)

        Pf = float(sum(cts))

        data = {
            'cfd_time': response['time'],
            'total_Fx': total_force['fx_avg'],
            'total_Fy': total_force['fy_avg'],
            'total_M': total_force['m_avg'],
            'Pf': Pf,
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

        for idx in range(self.wing_count):
            wing_no = idx + 1
            data.update({
                f'theta{wing_no}': motion[idx]['angle'],
                f'omega{wing_no}': motion[idx]['omega'],
                f'alpha{wing_no}': motion[idx]['omgac'],
                f'sa{wing_no}': motion[idx]['sa'],
                f'sau{wing_no}': motion[idx]['sau'],
                f'saac{wing_no}': motion[idx]['saac'],
                f'sb{wing_no}': motion[idx]['sb'],
                f'sbv{wing_no}': motion[idx]['sbv'],
                f'sbac{wing_no}': motion[idx]['sbac'],
                f'v_x{wing_no}': motion[idx]['sau'],
                f'v_y{wing_no}': motion[idx]['sbv'],
                f'a_x{wing_no}': motion[idx]['saac'],
                f'a_y{wing_no}': motion[idx]['sbac'],
                f'Fx{wing_no}': fxs[idx],
                f'Fy{wing_no}': fys[idx],
                f'M{wing_no}': torques[idx],
                f'T{wing_no}': fxs[idx],
                f'Ct{wing_no}': cts[idx],
                f'P{wing_no}': powers[idx],
                f'Cp{wing_no}': cps[idx],
                f'Pf{wing_no}': pfs[idx],
                f'Cl{wing_no}': cls[idx],
                f'control_w{wing_no}_domega': control_values[3 * idx],
                f'control_w{wing_no}_dsau': control_values[3 * idx + 1],
                f'control_w{wing_no}_dsbv': control_values[3 * idx + 2],
            })

        forces = [np.array([fxs[idx], fys[idx]]) for idx in range(self.wing_count)]
        zero_force = np.zeros(2)
        return {
            "forces": forces,
            "cls": cls,
            "cts": cts,
            "cps": cps,
            "powers": powers,
            "forces1": forces[0] if self.wing_count >= 1 else zero_force,
            "cl1": cls[0] if self.wing_count >= 1 else 0.0,
            "ct1": cts[0] if self.wing_count >= 1 else 0.0,
            "cp1": cps[0] if self.wing_count >= 1 else 0.0,
            "forces2": forces[1] if self.wing_count >= 2 else zero_force,
            "cl2": cls[1] if self.wing_count >= 2 else 0.0,
            "ct2": cts[1] if self.wing_count >= 2 else 0.0,
            "cp2": cps[1] if self.wing_count >= 2 else 0.0,
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
