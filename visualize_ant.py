import time
import gymnasium as gym
from stable_baselines3 import PPO

# 학습된 모델 파일 경로
model_path = "/media/leeseungmin/mydisk1/mujoco_logs/ppo_ant.zip"

# 렌더링 가능한 환경 생성 (render_mode 지정 필수)
eval_env = gym.make("Ant-v5", render_mode="human")
obs, _ = eval_env.reset()

# 학습된 모델 불러오기
model = PPO.load(model_path, env=eval_env)

try:
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)  # 결정론적 행동
        obs, reward, terminated, truncated, info = eval_env.step(action)
        time.sleep(0.01)  # 너무 빠른 렌더링 방지용 딜레이

        if terminated or truncated:
            obs, _ = eval_env.reset()

finally:
    eval_env.close()  # 반드시 환경 종료 (GLFW 오류 방지)
