import numpy as np


# Grid world parameters
rows = 3
cols = 4
gamma = 0.9        # Discount factor
epsilon = 1e-6       # Convergence threshold


# Define grid environment

#  ↓ X
#  0   [ ] [ ] [ ] [+1]    ← Terminal state with reward +1
#  1   [ ] [W] [ ] [-1]    ← Wall at (1,1), Terminal -1 at (1,3)
#  2   [S] [ ] [ ] [ ]     ← Start at (2,0)
#       0   1   2   3  → Y

# Terminal states, walls, and rewards
terminal_states = {
    (0, 3): 1.0,
    (1, 3): -1.0
}
walls = {(1, 1)}
non_terminal_reward = -0.04
start_state = (2, 0) # althoght the start state is not used in the algorithms, it's defined for completeness

# Actions and directions
actions = ['U', 'D', 'L', 'R']
action_symbols = {
    'U': '↑',
    'D': '↓',
    'L': '←',
    'R': '→'
}
action_vectors = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# Probabilities for intended and side-effect actions (stochastic movement)
action_probs = {
    'intended': 0.8,
    'left': 0.1,
    'right': 0.1
}

# Maps each action to its stochastic side effects: 10% chance to turn left or right
action_left = {'U': 'L', 'L': 'D', 'D': 'R', 'R': 'U'}
action_right = {'U': 'R', 'R': 'D', 'D': 'L', 'L': 'U'}

# Define initial policy for Policy Iteration
initial_policy = {
    (2, 0): 'U',
    (2, 1): 'L',
    (2, 2): 'U',
    (2, 3): 'L',
    (1, 0): 'R',
    (1, 2): 'D',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'D',
}

# All states in the grid world
states = []
for i in range(rows):
    for j in range(cols):
        s = (i, j)
        if s not in walls and s not in terminal_states:
            states.append(s)
# print("States:", states)

# Initialize the policy
policy = {}
for s in states:
    if s in initial_policy:
        policy[s] = initial_policy[s]
    else:
        policy[s] = np.random.choice(actions)  # Assign a random action for unspecified states

# Initialize the state-utility function U(s) to zero, including terminal states
U = {}
for s in states:
    U[s] = 0.0
for s in terminal_states:
    U[s] = terminal_states[s]  # Terminal states have no future value
# print(U)


def is_valid_state(s):
    """Check if the state is within the grid and not an obstacle."""
    i, j = s
    return 0 <= i < rows and 0 <= j < cols and s not in walls


def get_transitions(s, a):
    """
    Get the list of possible transitions from state s when action a is taken.
    Returns a list of tuples: (probability, next_state, reward)
    """
    transitions = []
    if s in terminal_states:
        # No transitions from terminal states
        return transitions
    
    # Intended action and unintended stochastic actions
    action_probabilities = [
        (a, action_probs['intended']),
        (action_left[a], action_probs['left']),
        (action_right[a], action_probs['right'])
    ]

    for action, prob in action_probabilities:
        di, dj = action_vectors[action]
        next_state = (s[0] + di, s[1] + dj)
        # Check for collisions with walls or obstacles
        if not is_valid_state(next_state):
            next_state = s  # Agent stays in the same state

        # Get the reward for the move
        if next_state in terminal_states:
            reward = terminal_states[next_state]  # Immediate reward upon entering terminal state
        else:
            reward = -0.04  # Standard reward for non-terminal moves

        transitions.append((prob, next_state, reward))

    return transitions


def print_policy(policy, iteration=None):
    """Print the policy grid."""
    if iteration is not None:
        print(f"\nPolicy after iteration {iteration}:")
    else:
        print("\nPolicy:")
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            if s in terminal_states:
                print(f" {terminal_states[s]:+} ", end='')
            elif s in walls:
                print(" XX ", end='')
            else:
                action = policy[s]
                print(f" {action_symbols[action]} ", end='')
        print()


def print_utility(U, iteration=None):
    """Print the utility-value grid."""
    if iteration is not None:
        print(f"\nUtility after iteration {iteration}:")
    else:
        print("\nUtility:")
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            if s in U:
                print(f"{U[s]:6.2f} ", end='')
            else:
                print("  XX   ", end='') # Placeholder for walls
        print()


# Policy Iteration Algorithm

is_policy_stable = False
iteration = 0
while not is_policy_stable:
    iteration += 1

    # Policy Evaluation
    while True:
        delta = 0
        for s in states:
            u = U[s]
            action = policy[s]
            transitions = get_transitions(s, action)
            U[s] = sum([
                prob * (reward + gamma * U[next_state])
                if next_state not in terminal_states else prob * reward
                for prob, next_state, reward in transitions
            ])
            delta = max(delta, abs(u - U[s]))

        if delta < epsilon:
            break

    # Policy Improvement
    is_policy_stable = True
    for s in states:
        old_action = policy[s]
        action_values = {}
        for a in actions:
            transitions = get_transitions(s, a)
            action_value = sum([
                prob * (reward + gamma * U[next_state])
                if next_state not in terminal_states else prob * reward
                for prob, next_state, reward in transitions
            ])
            action_values[a] = action_value
        best_action = max(action_values, key=action_values.get)
        policy[s] = best_action
        if old_action != best_action:
            is_policy_stable = False

    # Print the policy and value function after each iteration
    print_policy(policy, iteration)
    print_utility(U, iteration)

# Final output
print("\nOptimal Policy:")
print_policy(policy)
print("\nFinal Utility:")
print_utility(U)
