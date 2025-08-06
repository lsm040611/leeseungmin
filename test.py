from stable_baselines3 import PPO
from unitree_a1_env import UnitreeA1Env

# 환경 초기화 (시각화 모드)
env = UnitreeA1Env(render_mode="human")
model = PPO.load("ppo_unitree_a1_walk", env=env)

obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
