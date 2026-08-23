import numpy as np

class WarehouseMDP:
    def __init__(self):
        self.height = 5
        self.width = 6
        self.start = (4, 0)
        self.walls = {(0,2),(0,3),(1,1),(1,3),(2,1),(2,2),(3,3),(3,4)}
        self.slippery_states = {(1,2),(2,0),(3,1),(3,2),(2,4)}
        self.terminal_states = {(0,5):10.0,(2,5):2.0,(4,5):-10.0}
        self.danger_states = {(1,4):-3.0,(2,3):-3.0,(4,3):-3.0}
        self.living_reward = -1.0
        self.gamma = 0.9
        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]
    def is_valid_state(self, state):
        r,c = state
        return 0 <= r < self.height and 0 <= c < self.width and (r,c) not in self.walls
    def states(self):
        return [(r,c) for r in range(self.height) for c in range(self.width) if self.is_valid_state((r,c))]
    def is_terminal(self, state):
        return state in self.terminal_states
    def get_reward(self, state):
        if state in self.terminal_states: return self.terminal_states[state]
        if state in self.danger_states: return self.danger_states[state]
        if self.is_valid_state(state): return self.living_reward
        return 0.0
    def get_transition_probs(self, state, action):
        if self.is_terminal(state): return [(state,1.0)]
        if action == (-1,0): left=(0,-1); right=(0,1)
        elif action == (1,0): left=(0,1); right=(0,-1)
        elif action == (0,-1): left=(1,0); right=(-1,0)
        else: left=(-1,0); right=(1,0)
        outcomes = [(action,0.60),(left,0.20),(right,0.20)] if state in self.slippery_states else [(action,0.90),(left,0.05),(right,0.05)]
        result = {}
        for delta,p in outcomes:
            ns = (state[0]+delta[0], state[1]+delta[1])
            if not self.is_valid_state(ns): ns = state
            result[ns] = result.get(ns,0.0) + p
        return list(result.items())

def expected_next_value(grid, state, action, V):
    return sum(p * V[s2] for s2, p in grid.get_transition_probs(state, action))

def value_iteration(grid, threshold=1e-4, max_iter=10000):
    V = {s: 0.0 for s in grid.states()}
    for it in range(1, max_iter + 1):
        new_V = V.copy(); delta = 0.0
        for s in grid.states():
            if grid.is_terminal(s):
                new_V[s] = grid.get_reward(s)
            else:
                best = -np.inf
                for a in grid.actions:
                    q = grid.get_reward(s) + grid.gamma * expected_next_value(grid, s, a, V)
                    if q > best:
                        best = q
                new_V[s] = best
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < threshold:
            return V, it
    return V, max_iter

def extract_policy(grid, V):
    policy = {}
    for s in grid.states():
        if grid.is_terminal(s):
            continue
        best_a = None; best_v = -np.inf
        for a in grid.actions:
            val = grid.get_reward(s) + grid.gamma * expected_next_value(grid, s, a, V)
            if val > best_v:
                best_v = val; best_a = a
        policy[s] = best_a
    return policy

def policy_evaluation(grid, policy, threshold=1e-4, max_iter=10000):
    V = {s: grid.get_reward(s) for s in grid.states()}
    for _ in range(max_iter):
        new_V = V.copy(); delta = 0.0
        for s in grid.states():
            if grid.is_terminal(s):
                new_V[s] = grid.get_reward(s)
            else:
                a = policy[s]
                new_V[s] = grid.get_reward(s) + grid.gamma * expected_next_value(grid, s, a, V)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < threshold:
            break
    return V

def policy_improvement(grid, V, policy=None):
    if policy is None:
        policy = {}
    new_policy = {}
    for s in grid.states():
        if grid.is_terminal(s):
            continue
        best_a = policy.get(s, grid.actions[0])
        best_v = grid.get_reward(s) + grid.gamma * expected_next_value(grid, s, best_a, V)
        for a in grid.actions:
            val = grid.get_reward(s) + grid.gamma * expected_next_value(grid, s, a, V)
            if val > best_v + 1e-12:
                best_v = val; best_a = a
        new_policy[s] = best_a
    return new_policy

def policy_iteration(grid, threshold=1e-4, max_iter=100):
    policy = {s: grid.actions[0] for s in grid.states() if not grid.is_terminal(s)}
    history = []
    for _ in range(max_iter):
        V = policy_evaluation(grid, policy, threshold)
        new_policy = policy_improvement(grid, V, policy)
        history.append(len(new_policy))
        if new_policy == policy:
            return new_policy, V, history
        policy = new_policy
    return policy, V, history

grid = WarehouseMDP()
S = grid.states()
for s in S:
    for a in grid.actions:
        transitions = grid.get_transition_probs(s, a)
        total = sum(p for _, p in transitions)
        assert abs(total - 1.0) < 1e-12, (s, a, transitions)
V_vi, n_vi = value_iteration(grid)
pi_vi = extract_policy(grid, V_vi)
pi_pi, V_pi, history = policy_iteration(grid)
print('states', len(S))
print('vi_iterations', n_vi)
print('same_policy', pi_vi == pi_pi)
print('start_action_vi', pi_vi.get(grid.start))
print('start_action_pi', pi_pi.get(grid.start))
