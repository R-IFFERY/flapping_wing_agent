"""
扑翼系统的可视化模块。
提供扑翼运动、流场和性能指标的实时绘制功能。
"""
import numpy as np
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from config.config import PHYSICS_CONFIG, ENV_CONFIG
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FlappingWingVisualizer:
    def __init__(self, save_dir, dof_mode='1dof', wing_count=2):
        self.save_dir = save_dir
        self.dof_mode = dof_mode
        self.wing_count = int(wing_count)
        self.max_displacement = PHYSICS_CONFIG['max_displacement']
        os.makedirs(save_dir, exist_ok=True)

        # 根据自由度模式初始化维度
        if dof_mode == '1dof':
            self.n_dim = 1  # 只有俯仰角
        elif dof_mode == '2dof':
            self.n_dim = 2  # 升沉和俯仰
        else:  # 3dof
            self.n_dim = 3  # 轴向、升沉和俯仰

        # 存储历史数据
        self.reset_episode_data()

    def reset_episode_data(self):
        """重置回合数据"""
        self.time_steps = []
        self.positions = []    # 存储位置信息
        self.velocities = []   # 存储速度信息
        self.accelerations = []  # 存储加速度信息
        self.cls = [[] for _ in range(self.wing_count)]
        self.cts = [[] for _ in range(self.wing_count)]
        self.cps = [[] for _ in range(self.wing_count)]
        self.total_cts = []
        self.stabs = []
        self.rewards = []     # 存储奖励值
        self.actions = []     # 存储动作值

    def update(self, state, flow_data, reward, step, action,
               wing_cts=None, wing_cps=None, wing_cls=None, total_ct=0.0, stab=0.0):
        """更新并存储数据"""
        self.time_steps.append(step)
        self.actions.append(action)

        if self.dof_mode == '3dof':
            block = 3 * self.wing_count
            current_state = {
                'position': state[:block],
                'velocity': state[block:2 * block],
                'acceleration': state[2 * block:3 * block],
            }
        elif self.dof_mode == '2dof':
            block = 2 * self.wing_count
            current_state = {
                'position': state[:block],
                'velocity': state[block:2 * block],
                'acceleration': state[2 * block:3 * block],
            }
        elif self.dof_mode == '1dof':
            block = self.wing_count
            current_state = {
                'position': state[:block],
                'velocity': state[block:2 * block],
                'acceleration': state[2 * block:3 * block],
            }

        # 存储解析后的状态
        self.positions.append(np.array(current_state['position'], dtype=float))
        self.velocities.append(np.array(current_state['velocity'], dtype=float))
        self.accelerations.append(np.array(current_state['acceleration'], dtype=float))
        wing_cts = self._pad_wing_values(wing_cts if wing_cts is not None else [0.0] * self.wing_count)
        wing_cps = self._pad_wing_values(wing_cps if wing_cps is not None else [0.0] * self.wing_count)
        wing_cls = self._pad_wing_values(wing_cls if wing_cls is not None else [0.0] * self.wing_count)
        for wing_idx in range(self.wing_count):
            self.cts[wing_idx].append(wing_cts[wing_idx])
            self.cps[wing_idx].append(wing_cps[wing_idx])
            self.cls[wing_idx].append(wing_cls[wing_idx])
        self.total_cts.append(total_ct)
        self.stabs.append(stab)
        self.rewards.append(reward)

    def save_episode_summary(self, episode):
        """保存回合总结数据"""
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        actions = np.array(self.actions)

        # 计算实际时间序列
        time_series = np.array(self.time_steps) * PHYSICS_CONFIG['action_interval']

        def plot_series(ax, values, labels, ylabel, title):
            for idx, label in enumerate(labels):
                ax.plot(time_series, values[:, idx], label=label)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
            ax.legend()

        pos_labels, vel_labels, _ = self._state_column_labels()

        # 第一行第一列：位置曲线
        ax1 = axes[0, 0]
        plot_series(ax1, positions, pos_labels, 'Position / angle', 'Position and Angle History')

        # 第一行第二列：速度曲线
        ax2 = axes[0, 1]
        plot_series(ax2, velocities, vel_labels, 'Velocity', 'Velocity History')

        # 第二行第一列：升力系数曲线
        ax3 = axes[1, 0]
        for wing_idx in range(self.wing_count):
            ax3.plot(time_series, self.cls[wing_idx], label=f'wing[{wing_idx}].Cl')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Cl')
        ax3.set_title('Lift Coefficient History')
        ax3.grid(True)
        ax3.legend()

        # 第二行第二列：推力系数曲线
        ax4 = axes[1, 1]
        for wing_idx in range(self.wing_count):
            ax4.plot(time_series, self.cts[wing_idx], label=f'wing[{wing_idx}].Ct')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Ct')
        ax4.set_title('Thrust Coefficient History')
        ax4.grid(True)
        ax4.legend()

        # 第三行第一列：功率系数曲线
        ax5 = axes[2, 0]
        for wing_idx in range(self.wing_count):
            ax5.plot(time_series, self.cps[wing_idx], label=f'wing[{wing_idx}].Cp')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Cp')
        ax5.set_title('Power Coefficient History')
        ax5.grid(True)
        ax5.legend()

        # 第三行第二列：动作曲线
        ax6 = axes[2, 1]
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        action_labels = self._action_column_labels(actions.shape[1])
        for idx, label in enumerate(action_labels):
            ax6.plot(time_series, actions[:, idx], label=label)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Action')
        ax6.set_title('Action History')
        ax6.grid(True)
        ax6.legend()

        # 第四行第一列：奖励曲线
        ax7 = axes[3, 0]
        ax7.plot(time_series, self.rewards, 'k-', label='reward')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Reward')
        ax7.set_title('Reward History')
        ax7.grid(True)
        ax7.legend()

        fig.delaxes(axes[3, 1])

        # 调整子图之间的间距
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'episode_{episode}_summary.png'))
        plt.close(fig)

        # 保存数据到CSV
        self._save_episode_data(episode, time_series)

    def _save_episode_data(self, episode, time_series):
        """保存回合数据到CSV。"""
        data = {
            'time': time_series,
        }
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        accelerations = np.array(self.accelerations)
        actions = np.array(self.actions)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        pos_labels, vel_labels, acc_labels = self._state_column_labels()
        for idx, label in enumerate(pos_labels):
            data[label] = positions[:, idx]
        for idx, label in enumerate(vel_labels):
            data[label] = velocities[:, idx]
        for idx, label in enumerate(acc_labels):
            data[label] = accelerations[:, idx]

        data.update({
            'total_ct': self.total_cts,
            'stab': self.stabs,
            'reward': self.rewards,
        })
        for wing_idx in range(self.wing_count):
            data[f'wing[{wing_idx}]_cl'] = self.cls[wing_idx]
            data[f'wing[{wing_idx}]_ct'] = self.cts[wing_idx]
            data[f'wing[{wing_idx}]_cp'] = self.cps[wing_idx]

        action_labels = self._action_column_labels(actions.shape[1])
        for idx, label in enumerate(action_labels):
            data[label] = actions[:, idx]

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_dir, f'episode_{episode}_data.csv'), index=False)

    def _state_column_labels(self):
        pos_labels = []
        vel_labels = []
        acc_labels = []
        if self.dof_mode == '1dof':
            for wing_idx in range(self.wing_count):
                pos_labels.append(f'wing[{wing_idx}]_theta')
                vel_labels.append(f'wing[{wing_idx}]_omega')
                acc_labels.append(f'wing[{wing_idx}]_alpha')
        elif self.dof_mode == '2dof':
            for wing_idx in range(self.wing_count):
                pos_labels.extend([f'wing[{wing_idx}]_y', f'wing[{wing_idx}]_theta'])
                vel_labels.extend([f'wing[{wing_idx}]_v_y', f'wing[{wing_idx}]_omega'])
                acc_labels.extend([f'wing[{wing_idx}]_a_y', f'wing[{wing_idx}]_alpha'])
        else:
            for wing_idx in range(self.wing_count):
                pos_labels.extend([f'wing[{wing_idx}]_x', f'wing[{wing_idx}]_y', f'wing[{wing_idx}]_theta'])
                vel_labels.extend([f'wing[{wing_idx}]_v_x', f'wing[{wing_idx}]_v_y', f'wing[{wing_idx}]_omega'])
                acc_labels.extend([f'wing[{wing_idx}]_a_x', f'wing[{wing_idx}]_a_y', f'wing[{wing_idx}]_alpha'])
        return pos_labels, vel_labels, acc_labels

    def _action_column_labels(self, action_dim):
        if action_dim == self.wing_count * 3:
            components = ['domega', 'dsau', 'dsbv']
        elif self.dof_mode == '1dof':
            components = ['domega']
        elif self.dof_mode == '2dof':
            motion_axis = ENV_CONFIG.get('cfd_zmq', {}).get('motion_axis_2dof', 'sb')
            heave_component = 'dsau' if motion_axis == 'sa' else 'dsbv'
            components = [heave_component, 'domega']
        else:
            components = ['domega', 'dsau', 'dsbv']

        labels = []
        for wing_idx in range(self.wing_count):
            for component in components:
                labels.append(f'wing[{wing_idx}]_{component}')
        if len(labels) != action_dim:
            labels = [f'action_{idx + 1}' for idx in range(action_dim)]
        return labels

    def _pad_wing_values(self, values):
        values = list(values)
        if len(values) < self.wing_count:
            values.extend([0.0] * (self.wing_count - len(values)))
        return [float(v) for v in values[:self.wing_count]]

    def save_best_episode_animation(self, episode):
        """为最佳回合生成动画"""
        if len(self.positions) == 0:
            return  # 如果没有数据则直接返回

        positions = np.array(self.positions)

        # 根据不同的自���度模式创建不同的可视化
        if self.dof_mode == '1dof':
            # 1DOF模式：展示角度随时间的变化
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.set_xlim(0, len(positions))
            ax.set_ylim(-np.pi/2, np.pi/2)
            ax.set_xlabel('Time step')
            ax.set_ylabel('Pitch angle (rad)')
            ax.set_title('Pitch Angle History')
            ax.grid(True)

            line, = ax.plot([], [], 'b-', lw=2, label='pitch angle')
            point, = ax.plot([], [], 'ro', label='current position')
            ax.legend()

            def init():
                """初始化动画"""
                line.set_data([], [])
                point.set_data([], [])
                return line, point

            def animate(i):
                """更新动画帧"""
                # 绘制轨迹
                line.set_data(range(i+1), positions[:i+1, 0])
                # 绘制当前位置
                point.set_data([i], [positions[i, 0]])
                return line, point

        else:
            # 2DOF/3DOF模式：展示在x-y平面上的运动轨迹
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)

            if self.dof_mode == '2dof':
                # 2DOF模式：使用时间步作为x坐标，y位移作为y坐标
                time_steps = np.arange(len(positions))
                ax.set_xlim(0, len(positions))
                ax.set_ylim(-self.max_displacement, self.max_displacement)
                ax.set_xlabel('Time step')
                ax.set_ylabel('Y position (m)')
                ax.set_title('Heave Trajectory')
            else:
                # 3DOF模式：使用实际的x-y坐标
                ax.set_xlim(-self.max_displacement, self.max_displacement)
                ax.set_ylim(-self.max_displacement, self.max_displacement)
                ax.set_xlabel('X position (m)')
                ax.set_ylabel('Y position (m)')
                ax.set_title('Wing Motion Trajectory')

            ax.grid(True)

            line, = ax.plot([], [], 'b-', lw=2, label='trajectory')
            point, = ax.plot([], [], 'ro', label='current position')
            ax.legend()

            def init():
                """初始化动画"""
                line.set_data([], [])
                point.set_data([], [])
                return line, point

            def animate(i):
                """更新动画帧"""
                if self.dof_mode == '2dof':
                    # 2DOF模式：x轴是时间步，y轴是升沉位置
                    x_data = time_steps[:i+1]
                    y_data = positions[:i+1, 0]  # 第一列是y位置
                    current_x = time_steps[i]
                    current_y = positions[i, 0]
                else:
                    # 3DOF模式：使用实际的x-y位置
                    x_data = positions[:i+1, 0]  # 第一列是x位置
                    y_data = positions[:i+1, 1]  # 第二列是y位置
                    current_x = positions[i, 0]
                    current_y = positions[i, 1]

                # 绘制轨迹
                line.set_data(x_data, y_data)
                # 绘制当前位置
                point.set_data([current_x], [current_y])
                return line, point

        # 创建动画
        frames = len(positions)
        if frames > 0:
            anim = FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=50,
                               blit=True)

            # 保存动画
            anim.save(os.path.join(self.save_dir, f'episode_{episode}_animation.gif'),
                     writer='pillow', fps=20)

        plt.close(fig)

    def plot_training_history(self, history_data):
        """
        绘制训练历史曲线
        Args:
            history_data: 包含训练历史的字典，应该包含以下键：
                - 'rewards': 每个episode的奖励
                - 'actor_losses': Actor网络的损失
                - 'critic_losses': Critic网络的损失
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
        episodes = range(1, len(history_data['rewards']) + 1)

        # 绘制奖励曲线
        ax1.plot(episodes, history_data['rewards'], 'b-', label='Episode Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards over Episodes')
        ax1.grid(True)
        ax1.legend()

        # 绘制平均推力系数曲线
        ax2.plot(episodes, history_data['mean_total_cts'], 'm-', label='Mean Total Ct')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Mean Total Ct')
        ax2.set_title('Mean Total Ct over Episodes')
        ax2.grid(True)
        ax2.legend()



        # 绘制Actor损失曲线
        ax3.plot(episodes, history_data['actor_losses'], 'r-', label='Actor Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Actor Loss')
        ax3.set_title('Actor Network Loss over Episodes')
        ax3.grid(True)
        ax3.legend()

        # 绘制Critic损失曲线
        ax4.plot(episodes, history_data['critic_losses'], 'g-', label='Critic Loss')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Critic Loss')
        ax4.set_title('Critic Network Loss over Episodes')
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()

        self.save_training_history_data(episodes,history_data)

    def save_training_history_data(self, episodes, history_data):
        """将训练历史数据保存到CSV文件"""
        df = pd.DataFrame({
            'episode': episodes,
            'reward': history_data['rewards'],
            'mean_total_ct': history_data['mean_total_cts'],
            'actor_loss': history_data['actor_losses'],
            'critic_loss': history_data['critic_losses']
        })
        df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        print(f"训练历史数据已保存到: {os.path.join(self.save_dir, 'training_history.csv')}")
