import random
from collections import deque
import torch
import torch.nn.functional as F

from nets_utils import DQN

def initialise_episode(env):
    """Initialise an episode with random state.

    Args:
        env: A pygame environment instance.

    Returns:
        state, done, terminated, t: Returns the initial state, and boolean values indicating completion and episode count of episode.
    """
    observation, _ = env.reset()
    state = torch.tensor(observation).float()
    done, terminated, t = False, False, 0
    return state, done, terminated, t

def step_episode(env, policy_net:DQN, state, eps:float, decay:float, episode:int):
    """Do a single step in an episode.

    Args:
        env (): An openai pygym environment instance
        policy_net: A DQN policy network
        state: _description_
        eps: value of epsilon in an eps-greedy policy
        decay: A decay parameter decaying epsilon according to the episode count
        episode: The episode count

    Returns:
        action, next_state, reward, done, terminated: Returns an action, next state, reward, and whether the episode is completed.
    """
    from nets_utils import epsilon_greedy

    # determine an epsilon-greedy action
    action = epsilon_greedy(eps*(decay**episode), policy_net, state)       

    # do a step in the environment; convert values to Torch.tensor     
    observation, reward, done, terminated, _ = env.step(action)
    reward, action = torch.tensor([reward]), torch.tensor([action])
    next_state = torch.tensor(observation).reshape(-1).float()
    return action, next_state, reward, done, terminated



class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

################################################################################
################################################################################