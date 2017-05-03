import abc
import collections
import six
import copy

from seqmodel.common_tuple import EnvTransitionTuple


@six.add_metaclass(abc.ABCMeta)
class EnvGenerator(object):

    @abc.abstractmethod
    def reset(self, **kwargs):
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
            action: update with seq length
            done: whether the episode has ended
            info: contains auxiliary information (for debugging)
        """
        raise NotImplementedError('Not implemented.')

    @abc.abstractmethod
    def pack_transitions(self, transitions):
        """
        Pack transitions into a single data tuple. Useful for policy gradient
        Args:
            transitions: a list of EnvTransitionTuple objects
        Returns:
            full_observations: a single data tuple containing all transitions
        """
        raise NotImplementedError('Not implemented.')

    @abc.abstractmethod
    def replace_weights(self, obs, new_weights):
        """
        Replace weights in the observations. Used for policy update
        Args:
            obs: a data tuple
            new_weights: new weights
        Returns:
            new_obs
        """
        raise NotImplementedError('Not implemented.')

    @abc.abstractmethod
    def replace_labels(self, obs, new_labels):
        """
        Replace lables in the observations. Used for value update
        Args:
            obs: a data tuple
            new_labels: new labels
        Returns:
            new_obs
        """
        raise NotImplementedError('Not implemented.')


class Env(object):
    """
    A wrapper of batch namedtuple and its generator, for efficiency
    """
    def __init__(self, generator, re_init=False):
        self._generator = generator
        self._re_init = re_init
        self._ref_state = None
        self._cur_obs = None
        self._transitions = []

    @property
    def transitions(self):
        return self._transitions

    @property
    def packed_transitions(self):
        packed_obs = self._generator.pack_transitions(
            self._ref_state, self._transitions)
        packed_rewards = [t.reward for t in self._transitions]
        return packed_obs, packed_rewards

    def reset(self, new_obs=True):
        """
        Reset the state of the environment and return new batch of observations
        Returns:
            initial_obs: a batch initial observations
        """
        self._transitions = []
        if new_obs:
            self._ref_state, self._init_obs = self._generator.reset(
                re_init=self._re_init)
            self._ref_state = copy.deepcopy(self._ref_state)
        self._cur_obs = copy.deepcopy(self._init_obs)
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
        new_obs, action, done, info = self._generator.step(
            self._cur_obs, action)
        reward = self._reward(action, new_obs, done)
        self._transitions.append(
            EnvTransitionTuple(self._cur_obs, action, reward))
        self._cur_obs = new_obs
        return new_obs, reward, done, info

    def restart(self, **kwargs):
        self._generator.init_batch(**kwargs)

    def create_transition_return(self, states, ret):
        return self._generator.replace_weights(states, ret)

    def create_transition_value(self, states, values):
        return self._generator.replace_labels(states, values)

    def _reward(self, action, new_obs, done):
        return [0.0 for _ in range(len(action))]
