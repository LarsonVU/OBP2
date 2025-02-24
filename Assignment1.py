import numpy as np
import random
import math

def prob_calculator(next_state, current_state, rates, buffers):
    if -1 in next_state or any(next_state[i]>= buffers[i] +2 for i in range(len(next_state))):
        return 0
    active_rates = []
    if current_state[0] < buffers[0] +1:
        active_rates.append(rates[0])
    if current_state[0] > 0 and current_state[1] < buffers[0] +1:
        active_rates.append(rates[1])
    if current_state[1] > 0:
        active_rates.append(rates[2])
 
    if tuple(a - b for a, b in zip(next_state, current_state)) == (1,0):
        return rates[0] /sum(active_rates)
    if tuple(a - b for a, b in zip(next_state, current_state)) == (-1,1):
        return rates[1] /sum(active_rates)
    if tuple(a - b for a, b in zip(next_state, current_state)) == (0,-1):
        return rates[2] /sum(active_rates)
    return 0


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


def iterate_prob_matrix(rates, buffers):

    P, access = create_prob_matrix(rates, buffers)

    pi = np.array([1/len(P) for _ in range(len(P))])

    tolerance = 1e-6
    max_iters = 100
    diff = float('inf')
    iters = 0

    # Iterative process
    while diff > tolerance and iters < max_iters:
        pi_next = pi.T.dot(P)
        diff = max(abs(pi_next[i] - pi[i]) for i in range(len(pi)))
        pi = pi_next
        iters += 1

    return pi, access

mu1 = 3
mu2 = 3
mu3 = 5
B1 = 0
B2 = 0

pi, access = iterate_prob_matrix([mu1, mu2, mu3], [B1, B2])
# Display stationary distribution
# State (i,j) indicates i people waiting to finish service in queue 2, j people waiting to finish service in queue 3
for i in range(B1 + 2):
    for j in range(B2 + 2):
        print(f"pi({i},{j}) = {pi[access[(i,j,0,0)][0]]}")

        
#throughput = mu1 (1- sum(B+1, i))
throughput_a = min(mu1 * (1- sum([pi[access[(B1+1,i,0,0)][0]] for i in range(B2 +2)])), mu2 * (1- sum([pi[access[(i,B2+1,0,0)][0]] for i in range(B1 +2)])), mu3)
print(throughput_a)

# Exercise b
def exp_time(rate):
    return -math.log(1.0 - random.random()) / rate

np.random.seed(42)

def run_sim(mu1, mu2, mu3, B1, B2, max_sim_time):
    next_event_time = 0 

    queue1 = 0
    queue2 = 0

    m1_done = exp_time(mu1)
    m2_done = m1_done + exp_time(mu2)
    m3_done = m2_done + exp_time(mu3)

    produced = 0    

    while next_event_time < max_sim_time:
        next_event_time = min(m1_done, m2_done, m3_done)
        
        if m1_done == next_event_time:
            if queue1 < B1:
                m1_done = next_event_time + exp_time(mu1)
            else:
                m1_done = m2_done + exp_time(mu1)

            queue1 += 1
            print("queue1", queue1)
        
        if m2_done == next_event_time:
            if queue2 < B2:
                if queue2 > 0:
                    m2_done = next_event_time + exp_time(mu2)
                else:
                    m2_done = m1_done + exp_time(mu2)
            else:
                if queue1 > 0:
                    m2_done = m3_done + exp_time(mu2)
                else: 
                    m2_done = max(m1_done, m3_done) + exp_time(mu2)
            queue1 -= 1
            queue2 += 1
            print("queue2", queue2)

        if m3_done == next_event_time:
            if queue2 > 0:
                m3_done = next_event_time + exp_time(mu3)
            else:
                m3_done = m2_done + exp_time(mu3)
            queue2 -= 1
            produced += 1
            print(next_event_time, "produced", produced)

    return produced / max_sim_time
        
throughput_b = run_sim(mu1, mu2, mu3, B1, B2, 10000)
print(throughput_b)

