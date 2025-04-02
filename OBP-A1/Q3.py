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

def calculate_active_machines(max_buffer_sizes, state):
    rates = [0,0,0]

    if state[0] != max_buffer_sizes[0] + 1:
        rates[0] = 1

    if state[0] >= 1 and state[1] != max_buffer_sizes[1] + 1:
        rates[1] = 1

    if state[1] >= 1:
        rates[2] = 1

    return rates

def run_loop(start_state, mus, max_buffer_sizes, max_runtime, warmup, det):
    current_time = 0
    state = start_state
    total = 0
    active_machines = calculate_active_machines(max_buffer_sizes, state)

    if active_machines[det - 1] == 1:
        det_time = current_time + 1/mus[det - 1]
    else: det_time = float('inf')

    while current_time < max_runtime + warmup:
        active_rates = [mu * active for active, mu in zip(active_machines, mus)]
        active_rates[det - 1] = 0

        if sum(active_rates) == 0:
            machine = det
            current_time = det_time

        else:
            expo_time = current_time + sample_next_time(active_rates)

            if det_time < expo_time:
                machine = det
                current_time = det_time

            else: machine = sample_machine(active_rates); current_time = expo_time

        if machine == 1:
            state[0] += 1
                
        if machine == 2:
            state[1] += 1
            state[0] -= 1

        if machine == 3:
            if current_time > warmup:
                total += 1

            state[1] -= 1

        new_active_machines = calculate_active_machines(max_buffer_sizes, state)

        if new_active_machines[det - 1] == 1 and (active_machines[det - 1] == 0 or machine == det):
            det_time = current_time + 1 / mus[det - 1]
        elif new_active_machines[det - 1] == 0: det_time = float('inf')
        
        active_machines = new_active_machines

    return total

def run_sim(mus, max_buffer_sizes, max_runtime, warmup, det):
    start_state = [0, 0]
    total = run_loop(start_state, mus, max_buffer_sizes, max_runtime, warmup, det)
    return total / max_runtime

if __name__ == '__main__':
    mus = [1, 1.1, 0.9]
    max_buffer_sizes = [5, 5]
    max_runtime = 100000
    warmup = 10000
    det = 1

    throughput = run_sim(mus, max_buffer_sizes, max_runtime, warmup, det)
    print(throughput)

    num_sims = 100
    results = []

    max_runtime = 10000
    warmup = 1000
    
    for i in range(1,4):
        results = []
        print("Deterministic Machine: ", i)
        for _ in range(num_sims):
            throughput = run_sim(mus, max_buffer_sizes, max_runtime, warmup, i)
            results.append(throughput)

        mean = sum(results)/num_sims
        std = (sum([(r-mean)**2 for r in results])/num_sims)**(0.5)

        # Compute the 95% confidence interval (t_999,0.975 = 1.962)
        lower = mean - 1.962 * std / num_sims**(0.5)
        upper = mean + 1.962 * std / num_sims**(0.5)

        # Print only the confidence interval
        print(f"[{lower:.4f}, {upper:.4f}]")
        print("Mean throughput C: ", f"{mean:.4f}")
        print("Standard Deviation: ", f"{std:.4f}")