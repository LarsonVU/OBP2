import numpy as np


def prob_calculator(next_state, current_state, rates, buffers):
    if -1 in next_state or any(next_state[i]>= buffers[i] +2 for i in range(len(next_state))):
        return 0
    if tuple(a - b for a, b in zip(next_state, current_state)) == (1,0):
        return calc_x_grow(current_state, rates, buffers)
    if tuple(a - b for a, b in zip(next_state, current_state)) == (-1,1):
        return calc_y_grow(current_state, rates, buffers)
    if tuple(a - b for a, b in zip(next_state, current_state)) == (0,-1):
        return calc_y_shrink(current_state, rates, buffers)
    return 0

def calc_x_grow(current_state, rates, buffers):
    if current_state[0] == 0:
        if current_state[1] == 0:
            return rates[0] / rates[0]
        else:
            return rates[0] / (rates[0] + rates[2])
    else:
        if current_state[0] == 0 or current_state[1] == buffers[1] + 1:
            return rates[0] / (rates[0] + rates[2])
        else: 
            return rates[0] / (rates[0] + rates[1] + rates[2])
        
def calc_y_grow(current_state, rates, buffers):
    if current_state[0] == buffers[0] +1:
        if current_state[1] == 0:
            return rates[1] / rates[1]
        else: 
            return rates[1] / (rates[1] + rates[2])
    elif current_state[1] == 0:
        return rates[1] / (rates[1] + rates[0])
    else:
        return rates[1] / (rates[0] + rates[1] + rates[2])

def calc_y_shrink(current_state, rates, buffers):
    if current_state[0] == buffers[0] + 1:
        if current_state[1] == buffers[1] + 1:
            return rates[2] / rates[2]
        else:
            return rates[2] / (rates[1] + rates[2])
    else:
        if current_state[0] == 0 or current_state[1] == buffers[1] + 1:
            return rates[2] / (rates[0] + rates[2])
        else: 
            return rates[2] / (rates[0] + rates[1] + rates[2])

def create_prob_matrix(rates, buffers):
    prob_matrix = np.zeros((buffers[0] + 2, buffers[1] + 2, buffers[0] + 2, buffers[1] + 2))
    prob_matrix = prob_matrix.reshape((buffers[0] + 2) * (buffers[1] + 2), (buffers[0] + 2) * (buffers[1] + 2))
    state_access = {}
    for i in range(buffers[0] +2):
        for j in range(buffers[1] + 2):
            for a in range(buffers[0] + 2):
                for b in range(buffers[1] + 2):
                    state_access[(i,j, a,b)] = (i * (buffers[1] + 2) + j, a * (buffers[1] + 2) + b)
                    prob_matrix[state_access[(i,j, a,b)]] = prob_calculator((a,b), (i,j), rates, buffers)
    return prob_matrix, state_access

mu1 = 1
mu2 = 2
mu3 = 3
B1 = 10
B2 = 10


matrix, access = create_prob_matrix([mu1, mu2, mu3], [B1, B2])
matrix_power = np.linalg.matrix_power(matrix, 100)
print(matrix.round(3))
print(matrix_power.round(3))
