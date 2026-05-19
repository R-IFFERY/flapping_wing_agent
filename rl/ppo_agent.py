"""
PPO（近端策略优化）算法实现。
包含 Actor（策略网络）和 Critic（价值网络），并处理策略更新。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from config.config import TRAINING_CONFIG

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor输出层（均值），不使用Tanh约束
        self.actor_mean = nn.Linear(hidden_dim, action_dim)

        # 初始化为较小的标准差以限制初始探索
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * -1.0)

        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(1.0))
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, state):
        # 处理状态输入
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
        # 确保状态是二维张量
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Actor前向传播
        actor_features = self.actor(state)
        action_mean = self.actor_mean(actor_features)
        # 限制标准差的范围
        action_log_std = torch.clamp(self.actor_log_std, min=-20.0, max=2.0)
        action_std = action_log_std.exp()

        # Critic前向传播
        value = self.critic(state)

        return action_mean, action_std, value

    def get_action(self, state, deterministic=False):
        """选择动作"""
        action_mean, action_std, value = self(state)
        
        if deterministic:
            # 确定性行为时直接使用均值
            action = action_mean
            # 使用线性缩放将动作映射到[-1,1]
            action = torch.tanh(action)  # 平滑映射到(-1,1)
            return action, None, value

        # 使用重参数化技巧采样动作
        dist = Normal(action_mean, action_std)
        # 在原始动作空间采样
        action = dist.rsample()

        # 计算原始动作空间的对数概率
        log_prob = dist.log_prob(action)

        # 应用tanh变换
        action_tanh = torch.tanh(action)

        # 计算tanh变换后的对数概率修正
        # log prob of tanh-gaussian is log prob of gaussian minus log of derivative of tanh
        # dtanh/dx = 1 - tanh^2(x)
        log_prob_correction = torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob - log_prob_correction

        # 对所有维度求和
        log_prob = log_prob.sum(-1)

        return action_tanh, log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, learning_rate=3e-4,
                 gamma=0.99, clip_epsilon=0.2, device=None):
        """PPO代理初始化"""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ac_model = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=learning_rate)

        # PPO超参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        # ��配置文件读取价值函数相关参���
        self.value_clip_epsilon = TRAINING_CONFIG.get('value_clip_epsilon', 0.4)  # 价值函数裁剪范围，默认为策略裁剪的2倍
        self.use_value_clip = TRAINING_CONFIG.get('use_value_clip', False)  # 是否使用价值函数裁剪
        self.gae_lambda = TRAINING_CONFIG['gae_lambda']
        self.value_coef = TRAINING_CONFIG['value_coef']
        self.entropy_coef = TRAINING_CONFIG['entropy_coef']

        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """选择动作并返回相关信息"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            action, log_prob, value = self.ac_model.get_action(state)

        return (
            action.squeeze().cpu().numpy(),
            log_prob.squeeze().cpu().numpy(),
            value.squeeze().cpu().numpy()
        )

    def store_transition(self, state, action, reward, next_state, log_prob, value, done):
        """存储一个时间步的轨迹"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        """更新策略和价值网络"""
        if len(self.states) == 0:
            return 0, 0

        # 将所有数据转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        # 计算GAE和回报
        with torch.no_grad():
            _, _, next_value = self.ac_model(
                torch.FloatTensor(self.next_states[-1]).to(self.device)
            )
            advantages = []
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_val = next_value
                else:
                    next_val = old_values[t + 1]

                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - old_values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae)

            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = advantages + old_values
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 将数据分成小批次
        batch_size = TRAINING_CONFIG['mini_batch_size']
        n_epochs = TRAINING_CONFIG['update_epochs']

        for _ in range(n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(states))

            for start_idx in range(0, len(states), batch_size):
                # 获取小批次数据
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]

                # 获取当前策略的动作分布
                action_mean, action_std, values = self.ac_model(batch_states)
                dist = Normal(action_mean, action_std)

                # 计算新的动作对数概率，需要考虑tanh变换的影响
                # 注意：batch_actions已经是tanh变换后的值
                # 我们需要计算原始动作空间的概率，然后应用变换修正
                arctanh_actions = torch.atanh(torch.clamp(batch_actions, -0.999999, 0.999999))
                new_log_probs = dist.log_prob(arctanh_actions)

                # 应用tanh变换的对数概率修正
                log_prob_correction = torch.log(1.0 - batch_actions.pow(2) + 1e-6)
                new_log_probs = new_log_probs - log_prob_correction

                # 对所有维度求和
                new_log_probs = new_log_probs.sum(-1)

                # 计算策略比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 计算策略损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失（带裁剪）
                if self.use_value_clip:
                    values_clipped = batch_old_values + torch.clamp(
                        values.squeeze() - batch_old_values,
                        -self.value_clip_epsilon,
                        self.value_clip_epsilon
                    )
                    v_loss1 = (values.squeeze() - batch_returns) ** 2
                    v_loss2 = (values_clipped - batch_returns) ** 2
                    critic_loss = 0.5 * self.value_coef * torch.max(v_loss1, v_loss2).mean()
                else:
                    critic_loss = 0.5 * self.value_coef * (values.squeeze() - batch_returns).pow(2).mean()

                # 计算熵损失
                entropy_loss = -self.entropy_coef * dist.entropy().mean()

                # 总损失
                loss = actor_loss + critic_loss + entropy_loss

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_model.parameters(), TRAINING_CONFIG['max_grad_norm'])
                self.optimizer.step()

        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []

        return actor_loss.item(), critic_loss.item()

    def to(self, device):
        """将模型迁移到指定设备"""
        self.device = device
        self.ac_model.to(device)
        return self
