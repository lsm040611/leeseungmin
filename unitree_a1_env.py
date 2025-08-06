import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class UnitreeA1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        # 모델 로드
        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_a1/scene.xml")
        self.data = mujoco.MjData(self.model)

        # 시뮬레이션 파라미터
        self.sim_steps = 5
        self.timestep = self.model.opt.timestep
        self.max_time = 5.0
        self.current_time = 0.0

        # 렌더러 설정
        self.render_mode = render_mode
        self.viewer = None
        self.width, self.height = 720, 480  # 화면 확대

        # 관찰 및 행동 공간
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu  # 12개 관절
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # 안정 자세 (hip 0.2, thigh -0.9, calf 1.5)
        self.base_angles = np.array([
            0.2, -0.9, 1.5,  # FR
            0.2, -0.9, 1.5,  # FL
            0.2, -0.9, 1.5,  # RR
            0.2, -0.9, 1.5   # RL
        ])

    # ----------------------- Reset -----------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 초기 위치/자세 설정
        self.data.qpos[:] = 0
        self.data.qpos[2] = 0.30  # 몸통 높이 낮춤
        self.data.qpos[7:19] = self.base_angles  # 다리 관절 초기화
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.current_time = 0.0
        return self._get_obs(), {}

    # ----------------------- Step -----------------------
    def step(self, action):
        # Target angles = base + scaled action
        target_angles = self.base_angles + 0.05 * action  # scale 낮춤

        # PD control
        kp = 15
        kd = 0.5

        qpos = self.data.qpos[7:19]  # 관절 위치 (12개)
        qvel = self.data.qvel[6:18]  # 관절 속도 (12개)
        torque = kp * (target_angles - qpos) - kd * qvel
        self.data.ctrl[:] = np.clip(torque, -2.0, 2.0)  # 토크 제한 확대

        # 시뮬레이션 스텝
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
            self.current_time += self.timestep

        # 관찰, 보상, 종료 여부 반환
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_fall()
        truncated = self.current_time > self.max_time

        return obs, reward, terminated, truncated, {}
  # ----------------------- Render -----------------------
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.Renderer(self.model, self.width, self.height)

        mujoco.mj_forward(self.model, self.data)
        self.viewer.update_scene(self.data, camera=-1)

        if self.render_mode == "rgb_array":
            return self.viewer.render()

        elif self.render_mode == "human":
            img = self.viewer.render()
            from matplotlib import pyplot as plt
            plt.imshow(img)
            plt.pause(0.001)

    # ----------------------- Close -----------------------
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ----------------------- Helpers -----------------------
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _compute_reward(self):
        # Forward speed reward (x축 속도)
        forward_reward = 1.0 * self.data.qvel[0]

        # Height reward
        z = self.data.qpos[2]
        height_reward = 1.0 if 0.25 < z < 0.35 else -1.0

        # Balance reward (roll/pitch)
        quat = self.data.qpos[3:7]
        roll = np.arctan2(2*(quat[0]*quat[1]+quat[2]*quat[3]),
                          1-2*(quat[1]**2+quat[2]**2))
        pitch = np.arcsin(2*(quat[0]*quat[2]-quat[3]*quat[1]))
        balance_reward = -3.0 * (abs(roll) + abs(pitch))

        # Jump penalty (z 속도)
        z_vel = abs(self.data.qvel[2])
        jump_penalty = -2.0 * z_vel

        # Energy penalty
        energy_penalty = -0.001 * np.sum(np.square(self.data.ctrl))

        return forward_reward + height_reward + balance_reward + jump_penalty + energy_penalty

    def _check_fall(self):
        return self.data.qpos[2] < 0.15
