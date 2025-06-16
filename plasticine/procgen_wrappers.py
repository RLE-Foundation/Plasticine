
from procgen import ProcgenEnv

import numpy as np
import gym


class ContinualProcgen:
    """
    A wrapper for the Procgen environment that allows for switching between levels and tasks.

    Args:
        env_ids (list): List of environment IDs.
        num_envs (int): Number of environments to create.
        mode (str): Mode of the environment ('level' or 'task').
        level_offset (int): Level offset for the environment.
        gamma (float): Discount factor for the environment.
        shuffle (bool): Whether to shuffle of the environment IDs.
    
    Returns:
        None
    """
    def __init__(self,
                 env_ids,
                 num_envs,
                 mode,
                 level_offset,
                 gamma,
                 shuffle=False
                 ) -> None:
        self.env_ids = env_ids
        self.num_envs = num_envs
        self.mode = mode
        self.level_offset = level_offset
        self.gamma = gamma
        self.shuffle = shuffle

        if mode == 'level':
            print("Level mode will select the first item in the `env_ids` list !!!")
            assert level_offset >= 0, "Levels offset must be non-negative!"
            self.env_id = env_ids[0]
            self.current_level = 0
            self.envs = self.build_env()

        elif mode == 'task':
            # shuffle the env_ids
            if self.shuffle:
                np.random.shuffle(self.env_ids)
            self.env_id = env_ids[0]
            self.num_tasks = len(env_ids)
            self.round_step = 0
            self.current_level = 200
            assert len(self.env_ids) >= 5, "Task mode must have multiple environments!"
            self.envs = self.build_env()

        else:
            raise NotImplementedError("Mode must be either 'level' or 'task'!")

        # copy the observation and action space from the envs
        self.observation_space = self.envs.observation_space["rgb"]
        self.action_space = self.envs.action_space
        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space

    def build_env(self):
        # env setup
        envs = ProcgenEnv(num_envs=self.num_envs, 
                          env_name=self.env_id, 
                          num_levels=1, 
                          start_level=self.current_level, 
                          distribution_mode="easy")
        envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space["rgb"]
        envs.is_vector_env = True
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        # if args.capture_video:
        #     envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
        envs = gym.wrappers.NormalizeReward(envs, gamma=self.gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

        return envs
    
    def reset(self):
        return self.envs.reset()
    
    def step(self, actions):
        return self.envs.step(actions)

    def shift(self):
        if self.mode == 'level':
            # for each round, increase the level by `levels_offset`
            self.current_level += self.level_offset
            self.envs = self.build_env()
        elif self.mode == 'task':
            # change the environment for each round
            self.env_id = self.env_ids[self.round_step % self.num_tasks]
            self.envs = self.build_env()
            self.round_step += 1
        else:
            raise NotImplementedError("Mode must be either 'level' or 'task'!")