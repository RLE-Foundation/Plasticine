from procgen import ProcgenEnv
import numpy as np
import gym
from typing import Optional, Tuple, Any


class ContinualProcgenIntraTask:
    """
    A wrapper for ProcgenEnv that switches levels within a task.
    
    This wrapper follows the logic from crl_procgen where each level
    is trained sequentially. The level switches are typically managed
    externally in the training loop, but this wrapper provides the
    infrastructure for level switching.
    
    The wrapper applies standard gym wrappers for observation transformation,
    reward normalization, and episode statistics recording.
    
    Args:
        env_name: Name of the procgen environment (e.g., 'starpilot', 'bigfish', 'coinrun')
        num_envs: Number of parallel environments (default: 1)
        start_level: Starting level index (default: 0)
        max_levels: Maximum number of levels to cycle through (default: 10)
        level_steps: Number of level increments per switch (default: 20)
        distribution_mode: Distribution mode for procgen ('easy', 'hard', 'extreme', 'memory') (default: 'easy')
        seed: Seed for level generation (default: 0)
        gamma: Discount factor for reward normalization (default: 0.99)
    """
    
    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        start_level: int = 0,
        max_levels: int = 10,
        level_steps: int = 20,
        distribution_mode: str = "easy",
        seed: int = 0,
        gamma: float = 0.99,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.start_level = start_level
        self.max_levels = max_levels
        self.level_steps = level_steps
        self.distribution_mode = distribution_mode
        self.seed = seed
        self.gamma = gamma

        # Validate distribution_mode
        valid_modes = ['easy', 'hard', 'extreme', 'memory']
        if distribution_mode not in valid_modes:
            raise ValueError(f"distribution_mode must be one of {valid_modes}, got {distribution_mode}")

        # Initialize level tracking
        self.current_level = start_level

        # Build the environment
        self.envs = self.build_env()
        
        # Copy observation and action spaces
        self.observation_space = self.envs.observation_space["rgb"]
        self.action_space = self.envs.action_space
        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space
    
    def build_env(self):
        """
        Build the procgen environment with standard wrappers.
        
        Returns:
            Wrapped ProcgenEnv with observation transformation, reward normalization, etc.
        """
        # Create base ProcgenEnv
        envs = ProcgenEnv(
            num_envs=self.num_envs, 
            env_name=self.env_name, 
            num_levels=1, 
            start_level=self.current_level, 
            distribution_mode=self.distribution_mode
        )
        
        # Extract RGB observations from dict
        envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
        
        # Set single observation and action spaces for compatibility
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space["rgb"]
        envs.is_vector_env = True
        
        # Record episode statistics
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        
        # Normalize and clip rewards
        envs = gym.wrappers.NormalizeReward(envs, gamma=self.gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

        return envs
    
    def switch(self):
        """
        Switch to the next level by incrementing the current level.
        
        Note: In the original crl_procgen design, level switching is typically done
        externally in the training loop after a certain number of steps.
        This method can be called manually or integrated into the training loop.
        
        The level is incremented by level_steps, and a new environment is built.
        """
        # Increment level
        self.current_level += self.level_steps
        
        # Check if we've exceeded max_levels (optional, for safety)
        if self.max_levels > 0 and self.current_level >= self.start_level + self.max_levels * self.level_steps:
            # Cycle back or stop - for now, we'll just continue incrementing
            # In practice, this is usually managed externally
            pass
        
        # Close old environment and build new one
        if hasattr(self, 'envs'):
            self.envs.close()
        
        # Build new environment with updated level
        self.envs = self.build_env()
        
        # Update spaces (they should be the same, but update for consistency)
        self.observation_space = self.envs.observation_space["rgb"]
        self.action_space = self.envs.action_space
        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        """
        return self.envs.reset()
    
    def step(self, actions: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Step the environment.
        
        Args:
            actions: Actions to take in the environment (can be array for vectorized envs)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.envs.step(actions)
    
    def get_current_level(self) -> int:
        """
        Get the current level index.
        
        Returns:
            Current level index
        """
        return self.current_level
    
    def get_max_levels(self) -> int:
        """
        Get the maximum number of levels.
        
        Returns:
            Maximum number of levels
        """
        return self.max_levels
    
    def get_level_steps(self) -> int:
        """
        Get the number of level increments per switch.
        
        Returns:
            Number of level steps per switch
        """
        return self.level_steps
    
    def get_distribution_mode(self) -> str:
        """
        Get the distribution mode.
        
        Returns:
            Distribution mode string
        """
        return self.distribution_mode
    
    def close(self):
        """
        Close the environment and free resources.
        """
        if hasattr(self, 'envs'):
            self.envs.close()