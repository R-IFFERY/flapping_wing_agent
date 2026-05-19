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
from config.config import PHYSICS_CONFIG
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FlappingWingVisualizer:
    def __init__(self, save_dir, dof_mode='1dof'):
        self.save_dir = save_dir
        self.dof_mode = dof_mode
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
        self.ct1s = []         # 存储推力系数1
        self.ct2s = []
        self.cp1s = []
        self.cp2s = []
        self.pfs = []
        self.stabs = []
        self.rewards = []     # 存储奖励值
        self.actions = []     # 存储动作值

    def update(self, state, flow_data, reward, step, action, ct1, ct2, cp1, cp2, pf, stab):
        """更新并存储数据"""
        self.time_steps.append(step)
        self.actions.append(action)

        # 3DOF双扑翼状态向量：[x_position, y_1, theta_1, y_2, theta_2, v_x, v_y1, omega_1, v_y2, omega_2, a_x, a_y1, alpha_1, a_y2, alpha_2]
        if self.dof_mode == '3dof':
            current_state = {
                'position': [state[0], state[1], state[2], state[3], state[4]],  # [x_position, y_1, theta_1, y_2, theta_2]
                'velocity': [state[5], state[6], state[7], state[8], state[9]],  # [v_x, v_y1, omega_1, v_y2, omega_2]
                'acceleration': [state[10], state[11], state[12], state[13], state[14]]  # [a_x, a_y1, alpha_1, a_y2, alpha_2]
            }
        # 2DOF双扑翼状态向量：[y1, theta1, y2, theta2, vy1, omega1, vy2, omega2, ay1, alpha1, ay2, alpha2]
        elif self.dof_mode == '2dof':
            current_state = {
                'position': [state[0], state[1], state[2], state[3]],  # [y1, theta1, y2, theta2]
                'velocity': [state[4], state[5], state[6], state[7]],  # [vy1, omega1, vy2, omega2]
                'acceleration': [state[8], state[9], state[10], state[11]]  # [ay1, alpha1, ay2, alpha2]
            }
        # 1DOF单扑翼状态向量：[theta1, theta2, omega1, omega2, alpha1, alpha2]
        elif self.dof_mode == '1dof':
            current_state = {
                'position': [state[0], state[1]],  # [theta1, theta2]
                'velocity': [state[2], state[3]],  # [omega1, omega2]
                'acceleration': [state[4], state[5]]  # [alpha1, alpha2]
            }

        # 存储解析后的状态
        self.positions.append(current_state['position'])
        self.velocities.append(current_state['velocity'])
        self.accelerations.append(current_state['acceleration'])
        self.ct1s.append(ct1)
        self.ct2s.append(ct2)
        self.cp1s.append(cp1)
        self.cp2s.append(cp2)
        self.pfs.append(pf)
        self.stabs.append(stab)
        self.rewards.append(reward)

    def save_episode_summary(self, episode):
        """保存回合总结数据"""
        # 创建图像，4x2布局
        fig = plt.figure(figsize=(15, 25))
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        accelerations = np.array(self.accelerations)
        actions = np.array(self.actions)

        # 计算实际时间序列
        time_series = np.array(self.time_steps) * PHYSICS_CONFIG['action_interval']

        # 位置历史
        ax1 = fig.add_subplot(421)
        if self.dof_mode == '1dof':
            ax1.plot(time_series, positions[:, 0], 'b-', label='pitch 1')
            ax1.plot(time_series, positions[:, 1], 'r-', label='pitch 2')
        elif self.dof_mode == '2dof':
            ax1.plot(time_series, positions[:, 0], 'g-', label='y_1 (front wing)')
            ax1.plot(time_series, positions[:, 1], 'b-', label='theta_1 (front wing)')
            ax1.plot(time_series, positions[:, 2], 'r-', label='y_2 (rear wing)')
            ax1.plot(time_series, positions[:, 3], 'm-', label='theta_2 (rear wing)')
        else:  # 3dof
            ax1.plot(time_series, positions[:, 0], 'r-', label='x_position')
            ax1.plot(time_series, positions[:, 1], 'g-', label='y_1 (front wing)')
            ax1.plot(time_series, positions[:, 2], 'b-', label='theta_1 (front wing)')
            ax1.plot(time_series, positions[:, 3], 'c-', label='y_2 (rear wing)')
            ax1.plot(time_series, positions[:, 4], 'm-', label='theta_2 (rear wing)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position / angle')
        ax1.set_title('Position and Angle History')
        ax1.grid(True)
        ax1.legend()

        # 速度历史
        ax2 = fig.add_subplot(423)
        if self.dof_mode == '1dof':
            ax2.plot(time_series, velocities[:, 0], 'b-', label='angular velocity 1')
            ax2.plot(time_series, velocities[:, 1], 'r-', label='angular velocity 2')
        elif self.dof_mode == '2dof':
            ax2.plot(time_series, velocities[:, 0], 'g-', label='v_y1 (front wing)')
            ax2.plot(time_series, velocities[:, 1], 'b-', label='omega_1 (front wing)')
            ax2.plot(time_series, velocities[:, 2], 'r-', label='v_y2 (rear wing)')
            ax2.plot(time_series, velocities[:, 3], 'm-', label='omega_2 (rear wing)')
        else:  # 3dof
            ax2.plot(time_series, velocities[:, 0], 'r-', label='v_x')
            ax2.plot(time_series, velocities[:, 1], 'g-', label='v_y1 (front wing)')
            ax2.plot(time_series, velocities[:, 2], 'b-', label='omega_1 (front wing)')
            ax2.plot(time_series, velocities[:, 3], 'c-', label='v_y2 (rear wing)')
            ax2.plot(time_series, velocities[:, 4], 'm-', label='omega_2 (rear wing)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity History')
        ax2.grid(True)
        ax2.legend()

        # 加速度历史
        ax3 = fig.add_subplot(425)
        if self.dof_mode == '1dof':
            ax3.plot(time_series, accelerations[:, 0], 'b-', label='angular acceleration 1')
            ax3.plot(time_series, accelerations[:, 1], 'r-', label='angular acceleration 2')
        elif self.dof_mode == '2dof':
            ax3.plot(time_series, accelerations[:, 0], 'g-', label='a_y1 (front wing)')
            ax3.plot(time_series, accelerations[:, 1], 'b-', label='alpha_1 (front wing)')
            ax3.plot(time_series, accelerations[:, 2], 'r-', label='a_y2 (rear wing)')
            ax3.plot(time_series, accelerations[:, 3], 'm-', label='alpha_2 (rear wing)')
        else:  # 3dof
            ax3.plot(time_series, accelerations[:, 0], 'r-', label='a_x')
            ax3.plot(time_series, accelerations[:, 1], 'g-', label='a_y1 (front wing)')
            ax3.plot(time_series, accelerations[:, 2], 'b-', label='alpha_1 (front wing)')
            ax3.plot(time_series, accelerations[:, 3], 'c-', label='a_y2 (rear wing)')
            ax3.plot(time_series, accelerations[:, 4], 'm-', label='alpha_2 (rear wing)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration')
        ax3.set_title('Acceleration History')
        ax3.grid(True)
        ax3.legend()

        # 绘制推力系数历史
        ax4 = fig.add_subplot(422)
        ax4.plot(time_series, self.ct1s, 'r-', label='front wing Ct')
        ax4.plot(time_series, self.ct2s, 'g-', label='rear wing Ct')
        ax4.plot(time_series, self.cp1s, 'b-', label='front wing Cp')
        ax4.plot(time_series, self.cp2s, 'c-', label='rear wing Cp')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Coefficient')
        ax4.set_title('Force and Power Coefficients')
        ax4.grid(True)
        ax4.legend()

        # 绘制奖励历史
        ax5 = fig.add_subplot(424)
        ax5.plot(time_series, self.rewards, 'k-', label='reward')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Reward')
        ax5.set_title('Reward History')
        ax5.grid(True)
        ax5.legend()

        # 绘制动作历史
        ax6 = fig.add_subplot(426)
        if actions.ndim == 2 and actions.shape[1] == 6:
            labels = ['w1_domega', 'w1_dsau', 'w1_dsbv', 'w2_domega', 'w2_dsau', 'w2_dsbv']
            for idx, label in enumerate(labels):
                ax6.plot(time_series, actions[:, idx], label=label)
        elif self.dof_mode == '1dof':
            ax6.plot(time_series, actions, 'b-', label='angular velocity increment')
        elif self.dof_mode == '2dof':
            ax6.plot(time_series, actions[:, 0], 'g-', label='heave velocity increment')
            ax6.plot(time_series, actions[:, 1], 'b-', label='angular velocity increment')
        else:  # 3dof
            ax6.plot(time_series, actions[:, 0], 'r-', label='inline velocity increment')
            ax6.plot(time_series, actions[:, 1], 'g-', label='heave velocity increment')
            ax6.plot(time_series, actions[:, 2], 'b-', label='angular velocity increment')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Action')
        ax6.set_title('Action History')
        ax6.grid(True)
        ax6.legend()

        # 稳定性历史
        ax7 = fig.add_subplot(427)
        ax7.plot(time_series, self.stabs, 'm-', label='lift coefficient fluctuation')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Stability metric')
        ax7.set_title('Stability History')
        ax7.grid(True)
        ax7.legend()

        # 推力效率历史
        ax8 = fig.add_subplot(428)
        ax8.plot(time_series, self.pfs, 'c-', label='propulsive efficiency')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Efficiency')
        ax8.set_title('Propulsive Efficiency History')
        ax8.grid(True)
        ax8.legend()

        # 调整子图之间的间距
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'episode_{episode}_summary.png'))
        plt.close(fig)

        # 保存数据到CSV
        self._save_episode_data(episode, time_series)

    def _save_episode_data(self, episode, time_series):
        """保存回合数据到CSV，支持双扑翼3DOF"""
        data = {
            'time': time_series,
        }
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        accelerations = np.array(self.accelerations)
        actions = np.array(self.actions)
        cfd_action_mode = actions.ndim == 2 and actions.shape[1] == 6
        # 3DOF双扑翼参数
        if self.dof_mode == '3dof':
            data.update({
                'x_position': positions[:, 0],
                'y1': positions[:, 1],
                'theta1': positions[:, 2],
                'y2': positions[:, 3],
                'theta2': positions[:, 4],
                'v_x': velocities[:, 0],
                'v_y1': velocities[:, 1],
                'omega1': velocities[:, 2],
                'v_y2': velocities[:, 3],
                'omega2': velocities[:, 4],
                'a_x': accelerations[:, 0],
                'a_y1': accelerations[:, 1],
                'alpha1': accelerations[:, 2],
                'a_y2': accelerations[:, 3],
                'alpha2': accelerations[:, 4],
                'ct1': self.ct1s,
                'ct2': self.ct2s,
                'cp1': self.cp1s,
                'cp2': self.cp2s,
                'pf': self.pfs,
                'stab': self.stabs,
                'reward': self.rewards
            })
            if not cfd_action_mode:
                data.update({
                    'action_x': actions[:, 0],
                    'action_y1': actions[:, 1],
                    'action_theta1': actions[:, 2],
                    'action_y2': actions[:, 3],
                    'action_theta2': actions[:, 4],
                })
        # 2DOF参数
        elif self.dof_mode == '2dof':
            data.update({
                'y_1': positions[:, 0],
                'theta_1': positions[:, 1],
                'y_2': positions[:, 2],
                'theta_2': positions[:, 3],
                'v_y1': velocities[:, 0],
                'omega_1': velocities[:, 1],
                'v_y2': velocities[:, 2],
                'omega_2': velocities[:, 3],
                'a_y1': accelerations[:, 0],
                'alpha_1': accelerations[:, 1],
                'a_y2': accelerations[:, 2],
                'alpha_2': accelerations[:, 3],
                'ct1': self.ct1s,
                'ct2': self.ct2s,
                'cp1': self.cp1s,
                'cp2': self.cp2s,
                'pf': self.pfs,
                'stab': self.stabs,
                'reward': self.rewards
            })
            if not cfd_action_mode:
                data.update({
                    'action_y1': actions[:, 0],
                    'action_theta1': actions[:, 1],
                    'action_y2': actions[:, 2],
                    'action_theta2': actions[:, 3],
                })
        # 1DOF参数
        elif self.dof_mode == '1dof':
            data.update({
                'theta_1': positions[:, 0],
                'theta_2': positions[:, 1],
                'omega_1': velocities[:, 0],
                'omega_2': velocities[:, 1],
                'alpha_1': accelerations[:, 0],
                'alpha_2': accelerations[:, 1],
                'ct1': self.ct1s,
                'ct2': self.ct2s,
                'cp1': self.cp1s,
                'cp2': self.cp2s,
                'pf': self.pfs,
                'stab': self.stabs,
                'reward': self.rewards
            })
            if not cfd_action_mode:
                data.update({
                    'action_theta1': actions[:, 0],
                    'action_theta2': actions[:, 1],
                })

        if cfd_action_mode:
            data.update({
                'action_w1_domega': actions[:, 0],
                'action_w1_dsau': actions[:, 1],
                'action_w1_dsbv': actions[:, 2],
                'action_w2_domega': actions[:, 3],
                'action_w2_dsau': actions[:, 4],
                'action_w2_dsbv': actions[:, 5],
            })

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_dir, f'episode_{episode}_data.csv'), index=False)

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
        ax2.plot(episodes, history_data['mean_cts'], 'm-', label='Mean Thrust Coefficient')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Mean Thrust Coefficient')
        ax2.set_title('Mean Thrust Coefficient over Episodes')
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
            'mean_ct': history_data['mean_cts'],
            'mean_pf': history_data.get('mean_pfs', []),
            'actor_loss': history_data['actor_losses'],
            'critic_loss': history_data['critic_losses']
        })
        df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        print(f"训练历史数据已保存到: {os.path.join(self.save_dir, 'training_history.csv')}")
