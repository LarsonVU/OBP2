import math
import random

def sample_exponential(rate):
    return -math.log(1.0 - random.random()) / rate

def sample_next_time(rates):
    total_rate = sum(rates)
    if total_rate == 0:
        print(rates)
    return sample_exponential(total_rate)

def sample_machine(rates):
    x = random.random()

    total_rate = sum(rates)
    probs = [rate / total_rate for rate in rates]
    thresholds = [0]

    for i, p in enumerate(probs):
        thresholds.append(thresholds[i] + p)

    index = 0

    while x > thresholds[index]:
        index += 1

    return index

def calculate_active_rates(mus, max_buffer_sizes, state):
    rates = [0,0,0]

    if state[0] != max_buffer_sizes[0] + 1:
        rates[0] = mus[0]

    if state[0] >= 1 and state[1] != max_buffer_sizes[1] + 1:
        rates[1] = mus[1]

    if state[1] >= 1:
        rates[2] = mus[2]

    return rates

def run_loop(start_state, mus, max_buffer_sizes, max_runtime):
    current_time = 0
    state = start_state
    active_rates = [mus[0], 0, 0]
    total = 0

    while current_time < max_runtime:        
        active_rates = calculate_active_rates(mus, max_buffer_sizes, state)
        next_time = current_time + sample_next_time(active_rates)

        current_time = next_time
        
        machine = sample_machine(active_rates)

        if machine == 1:
            state[0] += 1

        if machine == 2:
            state[1] += 1
            state[0] -= 1

        if machine == 3:
            total += 1
            state[1] -= 1
                
    return state, total

def run_sim(mus, max_buffer_sizes, max_runtime, warmup):
    start_state = [0, 0]
    state, _ = run_loop(start_state, mus, max_buffer_sizes, warmup)
    state, total = run_loop(state, mus, max_buffer_sizes, max_runtime)
    return total / max_runtime

if __name__ == '__main__':
    mus = [1, 1.1, 0.9]
    max_buffer_sizes = [5, 5]
    max_runtime = 100000
    warmup = 10000

    throughput = run_sim(mus, max_buffer_sizes, max_runtime, warmup)
    print(throughput)

    num_sims = 1000
    results = []

    max_runtime = 10000
    warmup = 1000
    
    for _ in range(num_sims):
        throughput = run_sim(mus, max_buffer_sizes, max_runtime, warmup)
        results.append(throughput)

    mean = sum(results)/num_sims
    std = (sum([(r-mean)**2 for r in results])/num_sims)**(0.5)

    # Compute the 95% confidence interval (t_999,0.975 = 1.962)
    lower = mean - 1.962 * std / num_sims**(0.5)
    upper = mean + 1.962 * std / num_sims**(0.5)

    # Print only the confidence interval
    print(f"[{lower:.4f}, {upper:.4f}]")
    print("Mean throughput B: ", f"{mean:.4f}")
    print("Standard Deviation: ", f"{std:.4f}")