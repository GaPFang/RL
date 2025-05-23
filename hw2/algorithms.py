import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.returns = [[] for _ in range(self.state_space)]

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            trajectory = []
            while True:
                next_state, reward, done = self.collect_data()
                trajectory.append((current_state, reward))
                current_state = next_state
                if done:
                    break
            G = 0
            for i in range(len(trajectory) - 1, -1, -1):
                state, reward = trajectory[i]
                G = self.discount_factor * G + reward
                if state not in [s for s, _ in trajectory[:i]]:
                    self.returns[state].append(G)
                    self.values[state] = np.mean(self.returns[state])

                
class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            while True:
                next_state, reward, done = self.collect_data()
                if done:
                    self.values[current_state] += self.lr * (reward - self.values[current_state])
                    current_state = next_state
                    break
                self.values[current_state] += self.lr * (reward + self.discount_factor * self.values[next_state] - self.values[current_state])
                current_state = next_state


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            T = np.inf
            t = 0
            rewards = [0]
            states = [current_state]
            while True:
                if t < T:
                    next_state, reward, done = self.collect_data()
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                tau = t - self.n + 1
                # print(tau, self.n, len(rewards))
                if tau >= 0:
                    G = sum([self.discount_factor ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + self.n, T) + 1)])
                    if tau + self.n < T:
                        G += self.discount_factor ** self.n * self.values[states[tau + self.n]]
                    self.values[states[tau]] += self.lr * (G - self.values[states[tau]])
                if tau == T - 1:
                    break
                t += 1

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def epsilon_greedy(self, state_trace, action_trace, reward_trace) -> None:
        """Epsilon-greedy policy for selecting action"""
        while True:
            current_state = state_trace[-1]
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                action = self.policy_index[current_state]
            next_state, reward, done = self.grid_world.step(action)
            state_trace.append(next_state)
            action_trace.append(action)
            reward_trace.append(reward)
            if done:
                break

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        G = 0
        while action_trace:
            state = state_trace.pop(-2)
            action = action_trace.pop(-1)
            reward = reward_trace.pop(-1)
            # print(state, action, reward)
            G = self.discount_factor * G + reward
            self.q_values[state, action] += self.lr * (G - self.q_values[state, action])
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        for s in range(self.state_space):
            self.policy[s] = 0
            self.policy[s, self.q_values[s].argmax()] = 1


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # if iter_episode % 1000 == 0:
            #     print(f"Episode: {iter_episode}")
            self.get_policy_index()
            self.epsilon_greedy(state_trace, action_trace, reward_trace)
            self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()
            iter_episode += 1


class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def epsilon_greedy(self, state) -> int:
        """Epsilon-greedy policy for selecting action"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.policy_index[state]

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        if is_done:
            self.q_values[s, a] += self.lr * (r - self.q_values[s, a])
        else:
            self.q_values[s, a] += self.lr * (r + self.discount_factor * self.q_values[s2, a2] - self.q_values[s, a])

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            self.get_policy_index()
            # if iter_episode % 1000 == 0:
            #     print(f"Episode: {iter_episode}")
            prev_a = None
            is_done = False
            prev_a = self.epsilon_greedy(current_state)
            while not is_done:
                next_state, reward, is_done = self.grid_world.step(prev_a)
                action = self.epsilon_greedy(next_state)
                self.policy_eval_improve(current_state, prev_a, reward, next_state, action, is_done)
                current_state = next_state
                prev_a = action
            iter_episode += 1
            

class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size
    
    def epsilon_greedy(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.policy_index[state]

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        return np.random.choice(len(self.buffer), self.sample_batch_size)

    def policy_eval_improve(self, B) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        for b in B:
            s, a, r, s2, d = self.buffer[b]
            if d:
                self.q_values[s, a] += self.lr * (r - self.q_values[s, a])
            else:
                self.q_values[s, a] += self.lr * (r + self.discount_factor * self.q_values[s2].max() - self.q_values[s, a])

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        i = 0
        current_state = self.grid_world.reset()
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            self.get_policy_index()
            # if iter_episode % 1000 == 0:
            #     print(f"Episode: {iter_episode}")
            prev_a = None
            is_done = False
            while not is_done:
                prev_a = self.epsilon_greedy(current_state)
                next_state, reward, is_done = self.grid_world.step(prev_a)
                self.add_buffer(current_state, prev_a, reward, next_state, is_done)
                i += 1
                B = []
                if i % self.update_frequency == 0:
                    B = self.sample_batch()
                    self.policy_eval_improve(B)
                current_state = next_state
            iter_episode += 1