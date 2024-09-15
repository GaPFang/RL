import numpy as np

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        raise NotImplementedError


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        raise NotImplementedError

    def evaluate(self):
        while True:
            delta = 0
            old_values = self.values.copy()
            for state in range(self.grid_world.get_state_space()):
                v = 0
                policy_weights = []
                for action in range(self.grid_world.get_action_space()):
                    next_state, reward, end = self.grid_world.step(state, action)
                    policy_weight = self.policy[state, action]
                    policy_weights.append(policy_weight)
                    if end:
                        v += policy_weight * reward
                    else:
                        v += policy_weight * (reward + self.discount_factor * old_values[next_state])
                self.values[state] = v / sum(policy_weights)
                delta = max(delta, abs(old_values[state] - self.values[state]))
            if delta < self.threshold:
                break


    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        self.evaluate()


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0
            old_values = self.values.copy()
            for state in range(self.grid_world.get_state_space()):
                next_state, reward, end = self.grid_world.step(state, self.policy[state])
                if end:
                    self.values[state] = reward
                else:
                    self.values[state] = reward + self.discount_factor * old_values[next_state]
                delta = max(delta, abs(old_values[state] - self.values[state]))
            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        old_policy = self.policy.copy()
        for state in range(self.grid_world.get_state_space()):
            _, _, end = self.grid_world.step(state, self.policy[state])
            if end:
                continue
            best_action = None
            best_q_value = float("-inf")
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, _ = self.grid_world.step(state, action)
                q_value = reward + self.discount_factor * self.values[next_state]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            self.policy[state] = best_action
        return np.sum(old_policy != self.policy)
            

    def run(self) -> None:
        """Run the algorithm until convergence"""
        while True:
            self.policy_evaluation()
            if self.policy_improvement() == 0:
                break


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0
            old_values = self.values.copy()
            for state in range(self.grid_world.get_state_space()):
                _, reward, end = self.grid_world.step(state, self.policy[state])
                if end:
                    self.values[state] = reward
                else:
                    self.values[state] = float("-inf")
                    for action in range(self.grid_world.get_action_space()):
                        next_state, reward, _ = self.grid_world.step(state, action)
                        self.values[state] = max(self.values[state], reward + self.discount_factor * old_values[next_state])
                delta = max(delta, abs(old_values[state] - self.values[state]))
            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            _, _, end = self.grid_world.step(state, self.policy[state])
            if end:
                continue
            best_action = None
            best_q_value = float("-inf")
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, _ = self.grid_world.step(state, action)
                q_value = reward + self.discount_factor * self.values[next_state]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            self.policy[state] = best_action

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        self.policy_evaluation()
        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
    
    def policy_evaluation(self, mode):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        if (mode == "in-place"):
            while True:
                delta = 0
                for state in range(self.grid_world.get_state_space()):
                    _, reward, end = self.grid_world.step(state, self.policy[state])
                    if end:
                        self.values[state] = reward
                    else:
                        old_value = self.values[state]
                        self.values[state] = float("-inf")
                        for action in range(self.grid_world.get_action_space()):
                            next_state, reward, _ = self.grid_world.step(state, action)
                            self.values[state] = max(self.values[state], reward + self.discount_factor * self.values[next_state])
                        delta = max(delta, abs(self.values[state] - old_value))
                if delta < self.threshold:
                    break
        elif (mode == "prioritized"):
            deltas = np.zeros(self.grid_world.get_state_space())
            while True:
                # sort states by delta (descending)
                states = np.argsort(deltas)[::-1]
                for state in states:
                    _, reward, end = self.grid_world.step(state, self.policy[state])
                    if end:
                        self.values[state] = reward
                    else:
                        old_value = self.values[state]
                        self.values[state] = float("-inf")
                        for action in range(self.grid_world.get_action_space()):
                            next_state, reward, _ = self.grid_world.step(state, action)
                            self.values[state] = max(self.values[state], reward + self.discount_factor * self.values[next_state])
                        deltas[state] = abs(self.values[state] - old_value)
                delta = max(deltas)
                if delta < self.threshold:
                    break
        elif (mode == "RTDP"):
            while True:
                update = False
                state = 0
                while True:
                    old_value = self.values[state]
                    self.values[state] = float("-inf")
                    best_next_state = None
                    flag = False
                    for action in range(self.grid_world.get_action_space()):
                        next_state, reward, end = self.grid_world.step(state, action)
                        if (end):
                            flag = True
                            self.values[state] = reward
                            break
                        if (next_state == state):
                            continue
                        if (reward + self.discount_factor * self.values[next_state] > self.values[state]):
                            self.values[state] = reward + self.discount_factor * self.values[next_state]
                            best_next_state = next_state
                    delta = abs(self.values[state] - old_value)
                    if (delta > self.threshold):
                        update = True
                    if (flag):
                        break
                    state = best_next_state
                if (not update):
                    break
        elif (mode == "prioritized_by_value"):
            deltas = np.zeros(self.grid_world.get_state_space())
            while True:
                # sort states by value (descending)
                states = np.argsort(self.values)[::-1]
                for state in states:
                    _, reward, end = self.grid_world.step(state, self.policy[state])
                    if end:
                        self.values[state] = reward
                    else:
                        old_value = self.values[state]
                        self.values[state] = float("-inf")
                        for action in range(self.grid_world.get_action_space()):
                            next_state, reward, _ = self.grid_world.step(state, action)
                            self.values[state] = max(self.values[state], reward + self.discount_factor * self.values[next_state])
                        deltas[state] = abs(self.values[state] - old_value)
                delta = max(deltas)
                if delta < self.threshold:
                    break
        
    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            _, _, end = self.grid_world.step(state, self.policy[state])
            if end:
                continue
            best_action = None
            best_q_value = float("-inf")
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, _ = self.grid_world.step(state, action)
                q_value = reward + self.discount_factor * self.values[next_state]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            self.policy[state] = best_action

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        self.policy_evaluation('prioritized_by_value')  # "in-place", "prioritized", "RTDP", "prioritized_by_value"
        self.policy_improvement()
