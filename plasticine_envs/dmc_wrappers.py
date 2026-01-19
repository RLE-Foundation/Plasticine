import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Any, List


class ContinualDMC:
    """
    A wrapper for DeepMind Control Suite environments that switches between tasks sequentially.
    
    This wrapper follows the logic from crl_dmc where different DMC tasks are trained
    sequentially in a continual learning setting. The wrapper manages task switching
    and applies standard gymnasium wrappers for normalization and reward clipping.
    
    Args:
        env_id: Task sequence identifier ('dog', 'walker', or 'quadruped')
        num_envs: Number of parallel environments (default: 1)
        gamma: Discount factor for reward normalization (default: 0.99)
        capture_video: Whether to capture videos (default: False)
        run_name: Name for video recording directory (default: None)
        seed: Random seed for environment initialization (default: None)
    """
    
    # Task sequences as defined in crl_dmc
    ENV_NAME_DICT = {
        'dog': [
            'dm_control/dog-stand-v0',
            'dm_control/dog-walk-v0',
            'dm_control/dog-run-v0',
            'dm_control/dog-trot-v0',
        ],
        'walker': [
            'dm_control/walker-stand-v0',
            'dm_control/walker-walk-v0',
            'dm_control/walker-run-v0',
        ],
        'quadruped': [
            'dm_control/quadruped-walk-v0',
            'dm_control/quadruped-run-v0',
            'dm_control/quadruped-walk-v0',  # Note: third task repeats the first
        ],
    }
    
    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        gamma: float = 0.99,
        capture_video: bool = False,
        run_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if env_id not in self.ENV_NAME_DICT:
            raise ValueError(f"env_id must be one of {list(self.ENV_NAME_DICT.keys())}, got {env_id}")
        
        self.env_id = env_id
        self.num_envs = num_envs
        self.gamma = gamma
        self.capture_video = capture_video
        self.run_name = run_name
        self.seed = seed
        
        # Get task sequence
        self.env_name_list = self.ENV_NAME_DICT[env_id]
        self.num_tasks = len(self.env_name_list)
        
        # Initialize task tracking
        self.current_task_idx = 0
        self.current_env_name = self.env_name_list[self.current_task_idx]
        
        # Build the current environment
        self.envs = self._build_env()
        
        # Copy observation and action spaces
        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
    
    def _make_env(self, env_name: str, idx: int):
        """
        Create a single environment with standard wrappers.
        
        This follows the make_env function from crl_dmc/utils/basic_utils.py
        """
        def thunk():
            if self.capture_video and idx == 0:
                env = gym.make(env_name, render_mode="rgb_array")
                if self.run_name:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
            else:
                env = gym.make(env_name)
            # Apply standard wrappers as in crl_dmc
            env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=self.gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            return env
        return thunk
    
    def _build_env(self):
        """
        Build the vectorized environment for the current task.
        """
        envs = gym.vector.SyncVectorEnv(
            [self._make_env(self.current_env_name, i) for i in range(self.num_envs)]
        )
        return envs
    
    def switch(self):
        """
        Switch to the next task in the sequence.
        
        Note: In the original crl_dmc design, task switching is typically done
        externally in the training loop after a certain number of timesteps.
        This method can be called manually or integrated into the training loop.
        """
        if self.current_task_idx < self.num_tasks - 1:
            self.current_task_idx += 1
        else:
            # Cycle back to start if all tasks are completed
            self.current_task_idx = 0
        
        self.current_env_name = self.env_name_list[self.current_task_idx]
        
        # Close old environment and build new one
        if hasattr(self, 'envs'):
            self.envs.close()
        self.envs = self._build_env()
        
        # Update spaces
        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        
        print(f"Switched to task {self.current_task_idx + 1}/{self.num_tasks}: {self.current_env_name}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        
        Args:
            seed: Optional seed for environment reset
            options: Optional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        reset_seed = seed if seed is not None else self.seed
        return self.envs.reset(seed=reset_seed, options=options)
    
    def step(self, action: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Step the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.envs.step(action)
    
    def get_current_task(self) -> str:
        """
        Get the current task name.
        
        Returns:
            Current task name
        """
        return self.current_env_name
    
    def get_current_task_idx(self) -> int:
        """
        Get the current task index.
        
        Returns:
            Current task index (0-based)
        """
        return self.current_task_idx
    
    def get_num_tasks(self) -> int:
        """
        Get the total number of tasks in the sequence.
        
        Returns:
            Total number of tasks
        """
        return self.num_tasks
    
    def get_task_list(self) -> List[str]:
        """
        Get the list of all task names in the sequence.
        
        Returns:
            List of task names
        """
        return self.env_name_list.copy()
    
    def close(self):
        """
        Close the environment.
        """
        if hasattr(self, 'envs'):
            self.envs.close()

