# inspired from https://github.com/denisyarats/dmc2gym

import logging
import os

import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium.core import Env
from gymnasium import spaces
from gymnasium.spaces import Box


def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DeepMindControl(Env):
    def __init__(
        self,
        env_ids,
        mode,
        seed,
        task_kwargs={},
        environment_kwargs={},
        rendering="egl",
        render_height=64,
        render_width=64,
        render_camera_id=0,
    ):
        """TODO comment up"""
        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering
        
        self.task_kwargs = task_kwargs
        self.environment_kwargs = environment_kwargs
        self.env_ids = env_ids
        self.mode = mode
        self.seed = seed
        
        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        if len(self.env_ids) == 1:
            assert mode == 'dynamic', "Dynamic mode only works with a single environment!"
            self.friction_number = 0
            self.env_id = env_ids[0]
            self.envs = self.build_env()
        elif mode == 'task':
            # shuffle the env_ids
            np.random.shuffle(self.env_ids)
            self.env_id = env_ids[0]
            self.num_tasks = len(env_ids)
            self.round_step = 0
            assert len(self.env_ids) >= 5, "Task mode must have multiple environments!"
            self.envs = self.build_env()
        else:
            raise NotImplementedError("Mode must be either 'dynamic' or 'task'!")

        

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1


    def build_env(self, xml_path=None):
        domain, task = self.env_id.split('_', 1)
        if domain == 'cup':
            domain = 'ball_in_cup'
        # env setup
        if xml_path is not None:
            self.environment_kwargs={'xml_path':xml_path}
        self._env = suite.load(
            domain,
            task,
            task_kwargs=self.task_kwargs,
            environment_kwargs=self.environment_kwargs
        )
        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])

        # set seed if provided with task_kwargs
        if "random" in self.task_kwargs:
            seed = self.task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)
        
        return self._env
        # self._observation_space = spaces.Dict(self._observation_space)
        
    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount, "final_observation": observation}
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, np.random.RandomState):
                seed = np.random.RandomState(seed)
            self._env.task._random = seed

        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info = {}
        return observation, info

    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width
        camera_id = camera_id or self.render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
    
    def shift(self, xml_path=None):
        if self.mode == 'dynamic':
            # for each round, change the dynamic
            self._env.close()
            self.envs = self.build_env(xml_path=xml_path)
        elif self.mode == 'task':
            self._env.close()
            # change the environment for each round
            self.env_id = self.env_ids[self.round_step % self.num_tasks]
            self.envs = self.build_env()
            self.round_step += 1
        else:
            raise NotImplementedError("Mode must be either 'level' or 'task'!")