"""
扑翼优化项目的主训练脚本。
"""
import os
import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from config.config import ENV_CONFIG, TRAINING_CONFIG, PHYSICS_CONFIG
from rl.environment import FlappingWingEnv
from rl.ppo_agent import PPOAgent
from visualization.plotter import FlappingWingVisualizer

def get_auto_device():
    """Return a usable torch device, avoiding CUDA builds that do not support the GPU arch."""
    if not torch.cuda.is_available():
        return 'cpu'

    try:
        major, minor = torch.cuda.get_device_capability(0)
        device_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and device_arch not in supported_arches:
            device_name = torch.cuda.get_device_name(0)
            print(
                f"当前 PyTorch 不支持 GPU {device_name} 的 CUDA 架构 {device_arch}，"
                f"已自动回退到 CPU。若要使用 GPU，请安装支持 {device_arch} 的 PyTorch。"
            )
            return 'cpu'
    except Exception as e:
        print(f"检测 CUDA 设备兼容性失败，已自动回退到 CPU：{e}")
        return 'cpu'

    return 'cuda'

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='扑翼优化训练程序')
    parser.add_argument('--dof_mode', type=str, default=ENV_CONFIG.get('dof_mode', '2dof'),
                      choices=['1dof', '2dof', '3dof'],
                      help='自由度模式')
    parser.add_argument('--wing_count', type=int, default=ENV_CONFIG.get('cfd_zmq', {}).get('wing_count', 2),
                      choices=[1, 2],
                      help='翼数量，可选 1 或 2；必须与 CFD 服务端配置一致')
    parser.add_argument('--episodes', type=int, default=700,
                      help='训练回合数')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda'],
                      help='运行设备；auto会在当前PyTorch不支持GPU架构时自动回退CPU')
    parser.add_argument('--load_model', type=str, default=None,
                      help='预训练模型加载路径')
    parser.add_argument('--cfd_host', type=str, default=None,
                      help='ZeroMQ CFD求解器主机')
    parser.add_argument('--cfd_port', type=int, default=None,
                      help='ZeroMQ CFD求解器端口')
    return parser.parse_args()

def main():
    # 使用基于当前时间的随机种子
    current_seed = int(time.time() * 1000) % 100000
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)

    # 解析命令行参数
    args = parse_args()
    ENV_CONFIG['dof_mode'] = args.dof_mode
    ENV_CONFIG.setdefault('cfd_zmq', {})['wing_count'] = args.wing_count
    if args.cfd_host is not None:
        ENV_CONFIG.setdefault('cfd_zmq', {})['host'] = args.cfd_host
    if args.cfd_port is not None:
        ENV_CONFIG.setdefault('cfd_zmq', {})['port'] = args.cfd_port
    TRAINING_CONFIG['num_episodes'] = args.episodes
    device = get_auto_device() if args.device == 'auto' else args.device

    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', f'{ENV_CONFIG["dof_mode"]}_{args.wing_count}wing_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # 初始化环境（传入保存路径）
    env = FlappingWingEnv(
        dof_mode=ENV_CONFIG['dof_mode'],
        save_dir=save_dir
    )
    # 从环境获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 打印初始化信息
    print(f"\n初始化信息:")
    print(f"自由度模式: {ENV_CONFIG['dof_mode']}")
    print(f"翼数量: {ENV_CONFIG['cfd_zmq']['wing_count']}")
    print(f"状态空间维度: {state_dim}")
    print(f"动作空间维度: {action_dim}")
    print("气动模式: zmq-cfd")
    print(f"设备: {device}\n")

    # 初始化智能体
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=TRAINING_CONFIG['hidden_dim'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        gamma=TRAINING_CONFIG['gamma'],
        clip_epsilon=TRAINING_CONFIG['clip_epsilon'],
        device=device
    )

    # 如果指定了预训练模型，加载它
    if args.load_model:
        agent.ac_model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"加载预训练模型: {args.load_model}")

    # 初始化可视化器
    visualizer = FlappingWingVisualizer(
        save_dir=save_dir,
        dof_mode=ENV_CONFIG['dof_mode'],
        wing_count=ENV_CONFIG['cfd_zmq']['wing_count'],
    )
    env.set_visualizer(visualizer)  # 设置环境的可视化器

    # 训练循环
    best_reward = float('-inf')
    best_episode = -1
    episode_rewards = []
    episode_pfs = []
    # 添加用于记录训练历史的列表
    history_data = {
        'rewards': [],
        'mean_cts': [],
        'mean_pfs': [],
        'actor_losses': [],
        'critic_losses': [],
    }

    try:
        for episode in range(TRAINING_CONFIG['num_episodes']):
            # 设置当前回合号
            env.set_episode(episode)
            visualizer.reset_episode_data()  # 重置可视化器的回合数据

            state = env.reset()
            episode_reward = 0
            episode_pf = []
            episode_ct = []

            # 记录初始状态，动作值为零
            visualizer.update(state=state, flow_data=None, reward=0, step=0, action=np.zeros(action_dim), ct1=0, ct2=0, cp1=0, cp2=0, pf=0, stab=0)

            while True:
                # 智能体选择动作并获取value值
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    action_mean, action_std, value = agent.ac_model(state_tensor)
                    action, log_prob, _ = agent.ac_model.get_action(state_tensor)
                    action = action.cpu().numpy().flatten()
                    value = value.cpu().numpy().flatten()[0]

                # 执行动作
                next_state, reward, done, info = env.step(action)

                # 存储经验（按照PPOAgent.store_transition的正确参数顺序）
                agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    log_prob=log_prob.item(),
                    value=value,
                    done=done
                )

                # 更新状态和统计信息
                state = next_state
                episode_reward += reward
                episode_pf.append(info['power_factor'])
                episode_ct.append(info['thrust_coefficient'])

                if done:
                    break

            # 更新智能体并获取损失值
            actor_loss, critic_loss = agent.update()

            # 记录本回合的数据
            mean_pf = np.mean(episode_pf)
            mean_ct = np.mean(episode_ct)
            episode_pfs.append(mean_pf)
            episode_rewards.append(episode_reward)

            # 记录训练历史数据
            history_data['rewards'].append(episode_reward)
            history_data['mean_cts'].append(mean_ct)
            history_data['mean_pfs'].append(mean_pf)
            history_data['actor_losses'].append(actor_loss)
            history_data['critic_losses'].append(critic_loss)

            # 保存回合总结图
            visualizer.save_episode_summary(episode)

            # 打印训练信息
            print(f"回合 {episode}/{TRAINING_CONFIG['num_episodes']} - 奖励: {episode_reward:.2f}, 平均推力效率: {mean_pf:.2f}")

            # 如果是最佳回合，保存模型和生成动画
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode = episode
                # 保存最佳模型
                torch.save(agent.ac_model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                # 生成动画
                # visualizer.save_best_episode_animation(episode)
    finally:
        env.close()

    # 打印训练总结
    print(f"\n训练完成！")
    print(f"最佳回合: {best_episode}")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最终平均推力效率: {episode_pfs[-1]:.2f}")

    # 绘制训练历史曲线
    visualizer.plot_training_history(history_data)

if __name__ == '__main__':
    main()
