from dm_control import suite
from dm_env import specs
from gymnasium.core import Env
from gymnasium.spaces import Box

import logging
import pickle
import os
import numpy as np
import gymnasium as gym
import xml.etree.ElementTree as ET


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


class StandardDMC(Env):
    # borrowed from https://github.com/imgeorgiev/dmc2gymnasium
    def __init__(
        self,
        domain,
        task,
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

        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

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

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
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


class ContinualDMC:
    """
    A wrapper for the DMC environment that allows for switching between dynamics and tasks.

    Args:
        env_ids (list): List of environment IDs.
        mode (str): Mode of the environment ('dynamics' or 'task').
        num_envs (int): Number of environments to run in parallel.
        seed (int): Random seed for the environment.
        capture_video (bool): Whether to capture video of the environment.
        run_name (str): Name of the run for video capture.
        shuffle (bool): Whether to shuffle of the environment IDs.

    Returns:
        None
    """
    def __init__(self,
                 env_ids,
                 mode,
                 num_envs=1,
                 seed=1,
                 capture_video=False, 
                 run_name='',
                 shuffle=False
                 ) -> None:
        self.env_ids = env_ids
        self.mode = mode
        self.num_envs = num_envs
        self.seed = seed
        self.capture_video = capture_video
        self.run_name = run_name
        self.shuffle = shuffle

        if mode == 'dynamics':
            self.env_ids = ['quadruped_walk']
            assert mode == 'dynamics', "Dynamics mode only works with a single environment!"
            self.current_env_id = env_ids[0]
            self.envs = self.build_env()

            self.friction_number = 0
            self.frictions = pickle.load(open("plasticine/frictions", "rb+"))

        elif mode == 'task':
            # shuffle the env_ids
            if self.shuffle:
                np.random.shuffle(self.env_ids)
            self.current_env_id = env_ids[0]
            self.num_tasks = len(env_ids)
            self.round_step = 0
            assert len(self.env_ids) >= 5, "Task mode must have multiple environments!"
            self.envs = self.build_env()
        
        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space
        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space
    
    def build_env(self, xml_path=None):
        def make_env(env_id, seed, idx, capture_video, run_name):
            env_kwargs = {'xml_path':xml_path} if xml_path is not None else {}
            def thunk():
                domain, task = env_id.split("_")
                if capture_video and idx == 0:
                    env = StandardDMC(domain, task, task_kwargs={"random": seed}, environment_kwargs=env_kwargs)
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
                else:
                    env = StandardDMC(domain, task, task_kwargs={"random": seed}, environment_kwargs=env_kwargs)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env

            return thunk
    
        envs = gym.vector.SyncVectorEnv(
            [make_env(self.current_env_id, self.seed + i, i, self.capture_video, self.run_name) 
             for i in range(self.num_envs)]
        )

        return envs

    def reset(self, seed):
        return self.envs.reset(seed=seed)

    def step(self, actions):
        return self.envs.step(actions)

    def shift(self):
        if self.mode == 'dynamics':
            # for each round, change the dynamic (coefficient of friction)
            old_file = os.path.join(os.path.dirname(suite.__file__), 'quadruped.xml')
            tree = ET.parse(old_file)
            root = tree.getroot()
            new_friction = self.frictions[self.seed][self.friction_number]
            self.friction_number += 1
            root[6][1][5][0].attrib['friction'] = str(new_friction)
            current_working_directory = os.getcwd()
            tree.write('quadruped.xml')
            xml_path = current_working_directory + '/quadruped.xml'
            self.envs = self.build_env(xml_path=xml_path)

        elif self.mode == 'task':
            # change the environment for each round
            self.env_id = self.env_ids[self.round_step % self.num_tasks]
            self.envs = self.build_env()
            self.round_step += 1
        else:
            raise NotImplementedError("Mode must be either 'dynamic' or 'task'!")
        
    def gen_xml(self, xml_path):
        """
        Generate the xml file for the environment.
        """
        pass