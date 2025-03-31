import streamlit as st
import numpy as np


st.title("System Reliability Configuration")

# User Inputs
epsilon = 10e-10
failure_rate = st.number_input("Failure rate of components", min_value=0.0, step=0.01, format="%.3f")
repair_rate = st.number_input("Repair rate", min_value=0.0, step=0.01, format="%.3f")
warm_standby = st.radio("Are unused components in warm stand-by?", ("Yes", "No"))
num_components = st.number_input("Total number of components", min_value=1, step=1)
num_required = st.number_input("Number of components needed for system function", min_value=1, max_value=num_components, step=1)
num_repairmen = st.number_input("Number of repairmen", min_value=1, step=1)


def balance_equations(n, s, failure_rate, repair_rate, warm_standby, k):
    lambda_mu_ratio = failure_rate / repair_rate

    if warm_standby == "Yes":
        # Warm standby case
        def pi_0_inverse():
            sum1 = sum([np.math.comb(n, j) * (lambda_mu_ratio ** j) for j in range(0, s + 1)])
            sum2 = sum([(np.math.factorial(n) / (np.math.factorial(n - j) * np.math.factorial(s) * (s ** (j - s)))) * (lambda_mu_ratio ** j) for j in range(s + 1, n + 1)])
            return sum1 + sum2

        pi_0 = 1 / pi_0_inverse()
        state_probs = [pi_0 * (np.math.comb(n, j) * (lambda_mu_ratio ** j)) if j <= s else
                       pi_0 * ((np.math.factorial(n) / (np.math.factorial(n - j) * np.math.factorial(s) * (s ** (j - s)))) * (lambda_mu_ratio ** j))
                       for j in range(0, n + 1)]
    else:
        # Cold standby case
        if s <= n - k+1:
            def pi_0_inverse():
                sum1 = sum([(k ** j) / np.math.factorial(j) * (lambda_mu_ratio ** j) for j in range(0, s + 1)])
                sum2 = sum([(k ** j) / (np.math.factorial(s) * (s ** (j - s))) * (lambda_mu_ratio ** j) for j in range(s + 1, n - k + 2)])
                return sum1 + sum2

            pi_0 = 1 / pi_0_inverse()
            state_probs = [pi_0 * ((k ** j) / np.math.factorial(j) * (lambda_mu_ratio ** j)) if j <= s else
                           pi_0 * ((k ** j) / (np.math.factorial(s) * (s ** (j - s))) * (lambda_mu_ratio ** j))
                           for j in range(0, n - k + 2)]
        else:
            def pi_0_inverse():
                return sum([(k ** j) / np.math.factorial(j) * (lambda_mu_ratio ** j) for j in range(0, n - k + 2)])

            pi_0 = 1 / pi_0_inverse()
            state_probs = [pi_0 * ((k ** j) / np.math.factorial(j) * (lambda_mu_ratio ** j)) for j in range(0, n - k + 2)]

    return state_probs



if st.button("Compute Up-Time Probability"):
    state_matrix = balance_equations(num_components, num_repairmen,failure_rate, repair_rate, warm_standby, num_required)
    up_time = sum([state_matrix[i] for i in range(0, num_components - num_required + 1)])
    st.write(f"### System Up-Time Probability: {up_time:.4f}")

component_cost = st.number_input("Cost of component", min_value=0.0, step=0.01, format="%.3f")
repair_cost = st.number_input("Cost of repair man", min_value=0.0, step=0.01, format="%.3f")
down_time_cost = st.number_input("Cost of down time", min_value=0.0, step=0.01, format="%.3f")

min_cost = float('inf')
optimal_config = None

if st.button("Compute Optimal Configuration"):
    for comp in range(1,11):
        for rep in range(1,11):
            state_matrix = balance_equations(comp, rep, failure_rate, repair_rate, warm_standby, num_required)
            up_time = sum([state_matrix[i] for i in range(0, num_components - num_required + 1)])
            cost = (comp * component_cost) + (rep * repair_cost) + ((1 - up_time) * down_time_cost)
            if cost < min_cost:
                min_cost = cost
                optimal_config = (comp, rep)
    st.write(f"### Optimal Configuration: {optimal_config[0]} components and {optimal_config[1]} repairmen")

