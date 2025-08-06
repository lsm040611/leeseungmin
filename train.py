from stable_baselines3 import PPO
from unitree_a1_env import UnitreeA1Env

# 환경 초기화 (학습 시 시각화 꺼두기)
env = UnitreeA1Env(render_mode=None)

# 모델 설정
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    learning_rate=3e-4,
    clip_range=0.2
)

# 학습
model.learn(total_timesteps=5000000)  # 필요 시 늘릴 수 있음
model.save("ppo_unitree_a1_walk")




