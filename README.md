# Flapping Wing Agent

基于 PPO 强化学习的双翼扑翼运动优化程序。程序可以使用内置准定常气动力模型运行，也可以通过 ZeroMQ 与 `LBM_Mutiwing` CFD 程序在线耦合，由 CFD 返回每个控制步的运动状态和气动力。

## 目录结构

```text
flapping_wing_agent/
├── main.py                  # 训练入口
├── config/config.py         # 物理参数、训练参数、CFD/ZMQ 配置
├── rl/
│   ├── environment.py       # Gymnasium 环境
│   ├── ppo_agent.py         # PPO Actor-Critic
│   └── reward.py            # 奖励函数
├── physics/
│   ├── aerodynamics.py      # 气动力接口与数据记录
│   ├── kinematics.py        # 准定常模式下的运动学模型
│   └── lbm_client.py        # ZeroMQ REQ 客户端
└── visualization/plotter.py # 训练曲线、回合图像输出
```

## 运行模式

### CFD 在线耦合模式

默认配置为 CFD 在线耦合模式：

```python
ENV_CONFIG = {
    'use_cfd': True,
    'cfd_zmq': {
        'host': '127.0.0.1',
        'port': 5555,
        'wing_count': 2,
    },
}
```

Python 程序作为 ZeroMQ `REQ` 客户端，连接 `LBM_Mutiwing` 的 `REP` 服务端。每个 episode 会先发送 `RESET 0`，之后按控制周期发送 `STEP`，并在结束时发送 `STOP`。

### 准定常模式

如果不使用 CFD，需要在 `config/config.py` 中改为：

```python
'use_cfd': False
```

然后直接运行 `python main.py` 即可。

## Python 依赖

程序依赖以下 Python 包：

```bash
pip install numpy pandas matplotlib torch gymnasium pyzmq
```

如果已经配置好 Python 环境，只需要确认这些包能被当前环境导入。

## 训练入口

基本命令：

```bash
python main.py
```

常用参数：

```bash
python main.py \
  --dof_mode 2dof \
  --episodes 700 \
  --device cpu \
  --cfd_host 127.0.0.1 \
  --cfd_port 5555
```

参数说明：

```text
--dof_mode     自由度模式，可选 1dof、2dof、3dof
--episodes     训练回合数
--device       运行设备，可选 auto、cpu、cuda
--load_model   加载已有模型参数
--cfd_host     CFD ZeroMQ 服务端主机
--cfd_port     CFD ZeroMQ 服务端端口
```

注意：如果 CFD 和 Python 程序在同一个计算节点运行，`--cfd_host 127.0.0.1` 是正确的。如果 CFD 在另一台节点，不能使用 `127.0.0.1`，需要改成 CFD 所在节点名或 IP。

## CFD 联合运行

先启动 CFD 服务端：

```bash
cd ../LBM_Mutiwing
export LBM_MOTION_SOURCE=ZMQ
export LBM_ZMQ_ENDPOINT=tcp://*:5555
mpirun -np 1 ./exe
```

另开一个同节点终端启动训练：

```bash
cd ../flapping_wing_agent
python main.py --dof_mode 2dof --episodes 700 --device cpu --cfd_host 127.0.0.1 --cfd_port 5555
```

也可以用一个脚本先在后台启动 CFD，再等待端口就绪后启动训练：

```bash
cd ../LBM_Mutiwing

export LBM_MOTION_SOURCE=ZMQ
export LBM_ZMQ_ENDPOINT=tcp://*:5555

mpirun -np 1 ./exe > lbm_zmq.log 2>&1 &
CFD_PID=$!

for i in $(seq 1 120); do
    if ss -ltn | grep -q ':5555 '; then
        echo "CFD ZMQ port 5555 is ready"
        break
    fi
    sleep 1
done

if ! ss -ltn | grep -q ':5555 '; then
    echo "CFD ZMQ port 5555 is not ready"
    tail -n 80 lbm_zmq.log
    kill "$CFD_PID" || true
    exit 1
fi

cd ../flapping_wing_agent

python main.py \
  --dof_mode 2dof \
  --episodes 700 \
  --device cpu \
  --cfd_host 127.0.0.1 \
  --cfd_port 5555

kill "$CFD_PID" || true
```

## 输出文件

每次训练会在 `results/` 下创建带时间戳的目录：

```text
results/<dof_mode>_<YYYYmmdd_HHMMSS>/
```

主要输出包括：

```text
best_model.pth                      # 当前训练中奖励最高的模型
aerodynamics_data/                  # 每个 episode 的气动数据 CSV
训练过程图像和回合统计图              # 由 visualization/plotter.py 输出
```

CFD 程序的日志会写入：

```text
../LBM_Mutiwing/lbm_zmq.log
```

## 常见问题

### CFD ZMQ port 5555 is not ready

含义：Python 训练程序等待 `5555` 端口超时，说明 `LBM_Mutiwing` 没有成功启动或没有绑定端口。

检查：

```bash
tail -n 80 ../LBM_Mutiwing/lbm_zmq.log
ss -ltn | grep 5555
```

常见原因：

```text
1. CFD 程序没有进入 ZMQ 模式，缺少 LBM_MOTION_SOURCE=ZMQ。
2. 端口被占用。
3. CFD 程序启动失败，缺少模块、库或 GPU 资源。
4. Python 和 CFD 不在同一个计算节点，但 Python 仍连接 127.0.0.1。
```

### Python 连接不上 CFD

确认两点：

```text
1. CFD 和 Python 是否在同一个计算节点。
2. 如果不在同一个节点，--cfd_host 不能写 127.0.0.1。
```

在同一个节点运行时推荐：

```bash
--cfd_host 127.0.0.1 --cfd_port 5555
```

## 关键配置

主要配置在 `config/config.py`：

```text
PHYSICS_CONFIG['action_interval']       RL 每次动作控制的物理时间间隔
PHYSICS_CONFIG['simulation_time']       每个 episode 的总物理时间
PHYSICS_CONFIG['max_velocity_increment']动作映射后的最大速度增量
TRAINING_CONFIG['num_episodes']         默认训练回合数
ENV_CONFIG['dof_mode']                  默认自由度模式
ENV_CONFIG['use_cfd']                   是否使用在线 CFD
ENV_CONFIG['cfd_zmq']['warmup_time']    每个 episode RESET 后的 CFD 预推进时间
```

## ZeroMQ 协议摘要

Python 客户端发送：

```text
RESET 0
STEP <target_time> <w1_domega> <w1_dsau> <w1_dsbv> <w2_domega> <w2_dsau> <w2_dsbv>
SAVE
STOP
```

CFD 服务端返回：

```text
FORCE ...
STATUS ...
ERROR ...
```

`physics/lbm_client.py` 负责解析这些回复并转换为训练环境使用的气动力、运动状态和总力矩数据。
