import streamlit as st
import numpy as np


st.title("System Reliability Configuration")

# User Inputs
epsilon = 10e-10
failure_rate = st.number_input("Failure rate of components", min_value=epsilon, step=0.01, format="%f")
repair_rate = st.number_input("Repair rate", min_value=epsilon, format="%f")
warm_standby = st.radio("Are unused components in warm stand-by?", ("Yes", "No"))
num_components = st.number_input("Total number of components", min_value=1, step=1)
num_required = st.number_input("Number of components needed for system function", min_value=1, max_value=num_components, step=1)
num_repairmen = st.number_input("Number of repairmen", min_value=1, step=1)


def prob_calculator(current_state, next_state, failure_rate, repair_rate, warm_standby, num_repairmen, num_components):
    down_rate = failure_rate * (current_state if warm_standby == "Yes" else min(current_state, 1))
    up_rate = repair_rate * min(num_repairmen, num_components - current_state)
    if up_rate == 0:
        inactive_rate = repair_rate
    elif down_rate == 0:
        inactive_rate = failure_rate
    else:   
        inactive_rate = 0

    total_rate = down_rate + up_rate + inactive_rate

    if next_state - current_state== 0:
        return inactive_rate / total_rate
    if next_state - current_state == -1:
        return down_rate /total_rate
    if  next_state - current_state == 1:
        return up_rate /total_rate
    return 0


def create_prob_matrix(failure_rate, repair_rate, warm_standby, num_repairmen, num_components):
    rows = num_components + 1
    prob_matrix = np.zeros((rows, rows))

    for i in range(rows):
        for j in range(rows):
                prob_matrix[i,j] = prob_calculator(i, j, failure_rate, repair_rate, warm_standby, num_repairmen, num_components)
    return prob_matrix


def iterate_prob_matrix(failure_rate, repair_rate, warm_standby, num_repairmen, num_components):
    P= create_prob_matrix(failure_rate, repair_rate, warm_standby, num_repairmen, num_components)
    pi = np.array([1/len(P) for _ in range(len(P))])

    tolerance = 1e-10
    max_iters = 10000
    diff = float('inf')
    iters = 0

    # Iterative process
    while diff > tolerance and iters < max_iters:
        pi_t_next = np.matmul(pi.T, P)
        diff = max(abs(pi_t_next[i] - pi[i]) for i in range(len(pi)))
        pi = pi_t_next
        iters += 1

    return pi

if st.button("Compute Up-Time Probability"):
    state_matrix = iterate_prob_matrix(failure_rate, repair_rate, warm_standby, num_repairmen, num_components)
    up_time = sum([state_matrix[i] for i in range(num_required, num_components + 1)])
    st.write(f"### System Up-Time Probability: {up_time:.4f}")


