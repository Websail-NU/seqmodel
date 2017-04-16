import abc
import collections
import six


@six.add_metaclass(abc.ABCMeta)
class EnvGenerator(object):

    @abc.abstractmethod
    def reset(self):
        """
        Reset the state of the environment and return new batch of observations
        Returns:
            initial_obs: a batch initial observations
            reference: a batch of fully rolled out states
        """
        raise NotImplementedError('Not implemented.')

    @abc.abstractmethod
    def step(self, observation, action):
        """
        Run one timestep of the environment
        Args:
            observation: a batch of observation before taking the action
            action: a batch of action
        Returns:
            observations: a new batch of observations from
                          the current environment
            done: whether the episode has ended
            info: contains auxiliary information (for debugging)
        """
        raise NotImplementedError('Not implemented.')


EnvTransitionTuple = collections.namedtuple("EnvTransitionTuple",
                                            ("state", "action", "reward"))


class Env(object):
    """
    A wrapper of batch namedtuple and its generator, for efficiency
    """
    def __init__(self, generator):
        self._generator = generator
        self._ref_state = None
        self._cur_obs = None
        self._transitions = []

    def reset(self):
        """
        Reset the state of the environment and return new batch of observations
        Returns:
            initial_obs: a batch initial observations
        """
        self._ref_state, self._cur_obs = self._generator.reset()
        return self._cur_obs

    def step(self, action):
        """
        Run one timestep of the environment
        Args:
            observation: a batch of observation before taking the action
            action: a batch of action
        Returns:
            observations: a new batch of observations from
                          the current environment
            reward: a batch reward of the action on the current observations
            done: a batch of boolean, whether the episode has ended
            info: contains auxiliary information (for debugging)
        """
        new_obs, done, info = self._generator.step(self._cur_obs, action)
        reward = self._reward(action, new_obs)
        self._transitions.append(
            EnvTransitionTuple(self._cur_obs, action, reward))
        return new_obs, reward, done, info

    def _reward(self, action, new_obs):
        return 0.0
