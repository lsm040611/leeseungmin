#학습시간을 늘렸더니 전보다 오래 걸어다녔음
import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

log_dir = "/media/leeseungmin/mydisk1/mujoco_logs/"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    env = gym.make("Ant-v5")
    env = Monitor(env, log_dir)
    return env

env = DummyVecEnv([make_env])  # 함수 리스트로 넣기 (make_env 함수 호출 안 함)

model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95
)

model.learn(total_timesteps=100000)
model.save(os.path.join(log_dir, "ppo_ant"))

eval_env = gym.make("Ant-v5", render_mode="human")
obs, _ = eval_env.reset()

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    time.sleep(0.01)
    if terminated or truncated:
        obs, _ = eval_env.reset()

eval_env.close()
model.save("ppo_ant_new")

