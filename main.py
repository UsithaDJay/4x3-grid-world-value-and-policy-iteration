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


# Value Iteration
def value_iteration():
    U = np.zeros((rows, cols))
    policy = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    # TODO: implement function
    return U, policy

# Policy Iteration
def policy_iteration():
    policy = [[np.random.choice(actions) if (r, c) not in terminal_states and (r, c) not in walls else ' '
               for c in range(cols)] for r in range(rows)]
    U = np.zeros((rows, cols))
    # TODO: implement function
    return U, policy

# Run both algorithms
U_vi, policy_vi = value_iteration()
U_pi, policy_pi = policy_iteration()


print("== Value Iteration ==")
print(U_vi, policy_vi)

print("\n== Policy Iteration ==")
print(U_pi, policy_pi)
