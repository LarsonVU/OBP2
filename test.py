import usefull_functions as uf

def prob_calculator(next_state, current_state, rates, buffer):
    active_rates = []
    if current_state < buffer + 1:
        active_rates.append(rates[0])
    if current_state > 0:
        active_rates.append(rates[1])
    
    if next_state - current_state == 1:
        return rates[0] / sum(active_rates)
    if next_state - current_state == -1:
        return rates[1] / sum(active_rates)
    return 0

def create_prob_matrix(rates, buffer):
    size = buffer + 2
    prob_matrix = [[0 for _ in range(size)] for _ in range(size)]
    state_access = {i: i for i in range(size)}
    
    for i in range(size):
        for j in range(size):
            prob_matrix[state_access[i]][state_access[j]] = prob_calculator(j, i, rates, buffer)
    
    return prob_matrix, state_access

def iterate_prob_matrix(rates, buffer):
    P, access = create_prob_matrix(rates, buffer)
    pi = [1 / len(P) for _ in range(len(P))]
    tolerance = 1e-10
    max_iters = 1000
    diff = float('inf')
    iters = 0
    
    while diff > tolerance and iters < max_iters:
        pi_next = uf.matmul([pi], P)[0]
        diff = max(abs(pi_next[i] - pi[i]) for i in range(len(pi)))
        pi = pi_next
        iters += 1
    
    return pi, access

mu1 = 100
mu2 = 0.9
B = 1

pi, access = iterate_prob_matrix([mu1, mu2], B)

for i in range(B + 2):
    print(f"pi({i}) = {pi[access[i]]}")

throughput = mu1 * (1 - pi[access[B + 1]])
print("Throughput:", throughput)

throughput2 = mu2 * (1- pi[access[0]])
print(throughput2)
