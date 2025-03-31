import streamlit as st
import numpy as np
import math


def balance_equations(n, s, failure_rate, repair_rate, warm_standby, k):
    lambda_mu_ratio = failure_rate / repair_rate

    if warm_standby == "Yes":
        # Warm standby case
        def pi_0_inverse():
            sum1 = sum([math.comb(n, j) * (lambda_mu_ratio ** j) for j in range(0, s + 1)])
            sum2 = sum([(math.factorial(n) / (math.factorial(n - j) * math.factorial(s) * (s ** (j - s)))) * (lambda_mu_ratio ** j) for j in range(s + 1, n + 1)])
            return sum1 + sum2

        pi_0 = 1 / pi_0_inverse()
        state_probs = [pi_0 * (math.comb(n, j) * (lambda_mu_ratio ** j)) if j <= s else
                       pi_0 * ((math.factorial(n) / (math.factorial(n - j) * math.factorial(s) * (s ** (j - s)))) * (lambda_mu_ratio ** j))
                       for j in range(0, n + 1)]
    else:
        # Cold standby case
        if s <= n - k+1:
            def pi_0_inverse():
                sum1 = sum([(k ** j) / math.factorial(j) * (lambda_mu_ratio ** j) for j in range(0, s + 1)])
                sum2 = sum([(k ** j) / (math.factorial(s) * (s ** (j - s))) * (lambda_mu_ratio ** j) for j in range(s + 1, n - k + 2)])
                return sum1 + sum2

            pi_0 = 1 / pi_0_inverse()
            state_probs = [pi_0 * ((k ** j) / math.factorial(j) * (lambda_mu_ratio ** j)) if j <= s else
                           pi_0 * ((k ** j) / (math.factorial(s) * (s ** (j - s))) * (lambda_mu_ratio ** j))
                           for j in range(0, n - k + 2)]
        else:
            def pi_0_inverse():
                return sum([(k ** j) / math.factorial(j) * (lambda_mu_ratio ** j) for j in range(0, n - k + 2)])

            pi_0 = 1 / pi_0_inverse()
            state_probs = [pi_0 * ((k ** j) / math.factorial(j) * (lambda_mu_ratio ** j)) for j in range(0, n - k + 2)]

    return state_probs

def compute_up_time(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required):
    state_matrix = balance_equations(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required)
    up_time = sum([state_matrix[i] for i in range(0, num_components - num_required + 1)])
    return up_time


st.title("System Reliability Configuration")

tab1, tab2 = st.tabs(["Exercise A", "Exercise B"])
with tab1:
    # User Inputs
    failure_rate = st.number_input("Failure rate of components", min_value=0.0, step=0.01, format="%.2f")
    repair_rate = st.number_input("Repair rate", min_value=0.0, step=0.01, format="%.2f")
    warm_standby = st.radio("Are unused components in warm stand-by?", ("Yes", "No"))
    num_components = st.number_input("Total number of components", min_value=1, step=1)
    num_required = st.number_input("Number of components needed for system function", min_value=1, max_value=num_components, step=1)
    num_repairmen = st.number_input("Number of repairmen", min_value=1, step=1)

    if st.button("Compute Up-Time Probability"):
        up_time = compute_up_time(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required)
        st.write(f"### System Up-Time Probability: {up_time:.4f}")

def compute_cost(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required, component_cost, repair_cost, down_time_cost):
    if num_components < num_required:
        up_time = 0
    else:
        up_time = compute_up_time(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required)
    return  (comp * component_cost) + (rep * repair_cost) + ((1 - up_time) * down_time_cost)


with tab2:
    failure_rate_b = st.number_input("Failure rate of components", min_value=0.0, step=0.01, key="fail_b")
    repair_rate_b = st.number_input("Repair rate", min_value=0.0, step=0.01, key="repair_b")
    warm_standby_b = st.radio("Are unused components in warm stand-by?", ("Yes", "No"), key="warm_standby_b")
    num_required_b = st.number_input("Number of components needed for system function", min_value=1, step=1, key="num_required_b")

    component_cost = st.number_input("Cost of component", min_value=0, step=1)
    repair_cost = st.number_input("Cost of repair man", min_value=0, step=1)
    down_time_cost = st.number_input("Cost of down time", min_value=0, step=1)

    min_cost = float('inf')
    optimal_config = None


    if st.button("Compute Optimal Configuration"):
        for comp in range(1,50):
            for rep in range(1,50): 
                cost = compute_cost(comp, rep, failure_rate_b, repair_rate_b, warm_standby_b, num_required_b, component_cost, repair_cost, down_time_cost)
                if cost < min_cost:
                    min_cost = cost
                    optimal_config = (comp, rep)

        st.write(f"### Optimal Configuration: {optimal_config[0]} components and {optimal_config[1]} repairmen")

