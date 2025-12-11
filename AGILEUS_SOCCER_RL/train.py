import gymnasium as gym
import torch
import torch.nn as nn

# Import SKRL components (You already installed these)
from skrl.models.torch import Model, GaussianDynamic
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

# 1. SETUP THE ENVIRONMENT (DeepMind Humanoid)
# We use the standard high-fidelity Humanoid-v4 task
env = gym.make('Humanoid-v4', render_mode=None)
env = wrap_env(env) # Wrap for SKRL compatibility

device = env.device

# 2. DEFINE THE BRAIN (Policy Network)
class Policy(Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        # A deep network for complex agility
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ELU(),
            nn.Linear(400, 300),
            nn.ELU(),
            nn.Linear(300, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter

# 3. DEFINE THE VALUE ESTIMATOR (Critic)
class Value(Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ELU(),
            nn.Linear(400, 300),
            nn.ELU(),
            nn.Linear(300, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"])

# 4. CONFIGURE & TRAIN
# Instantiate the models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device)
models["value"] = Value(env.observation_space, env.action_space, device)

# Configure PPO (Hyperparameters for Locomotion)
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"]["num_rollouts"] = 1024  # Scale of training
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["grad_norm_clip"] = 1.0
cfg["state_preprocessor"] = True  # Normalize inputs for stable training

agent = PPO(models=models,
            memory=RandomMemory(memory_size=4096, num_envs=env.num_envs, device=device),
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# 5. EXECUTE TRAINING
print(f"--- STARTING AGILEUS (MuJoCo) TRAINING ON {device} ---")
trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent)
trainer.train()