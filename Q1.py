import usefull_functions as uf

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
    rows = (buffers[0] + 2) * (buffers[1] + 2)
    cols = (buffers[0] + 2) * (buffers[1] + 2)
    prob_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    state_access = {}
    for i in range(buffers[0] +2):
        for j in range(buffers[1] + 2):
            for a in range(buffers[0] + 2):
                for b in range(buffers[1] + 2):
                    state_access[(i,j, a,b)] = (i * (buffers[1] + 2) + j, a * (buffers[1] + 2) + b)
                    prob_matrix[state_access[(i,j, a,b)][0]][state_access[(i,j, a,b)][1]] = prob_calculator((a,b), (i,j), rates, buffers)
    return prob_matrix, state_access


def iterate_prob_matrix(rates, buffers):

    P, access = create_prob_matrix(rates, buffers)

    pi = [1/len(P) for _ in range(len(P))]

    tolerance = 1e-6
    max_iters = 100
    diff = float('inf')
    iters = 0

    # Iterative process
    while diff > tolerance and iters < max_iters:
        pi_next = uf.matmul([pi], P)[0]
        diff = max(abs(pi_next[i] - pi[i]) for i in range(len(pi)))
        pi = pi_next
        iters += 1

    return pi, access

mu1 = 1
mu2 = 1.1
mu3 = 0.9
B1 = 5
B2 = 5

pi, access = iterate_prob_matrix([mu1, mu2, mu3], [B1, B2])
# Display stationary distribution
# State (i,j) indicates i people waiting to finish service in queue 2, j people waiting to finish service in queue 3
for i in range(B1 + 2):
    for j in range(B2 + 2):
        print(f"pi({i},{j}) = {pi[access[(i,j,0,0)][0]]}")

        
#throughput = mu1 (1- sum(B+1, i))
throughput_a = min(mu1 * (1- sum([pi[access[(B1+1,i,0,0)][0]] for i in range(B2 +2)])), mu2 * (1- sum([pi[access[(i,B2+1,0,0)][0]] for i in range(B1 +2)])), mu3)
print(throughput_a)
