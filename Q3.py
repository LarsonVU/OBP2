from usefull_functions import sample_exponential

def run_sim_with_deterministic(mus, max_buffer_lengths, run_time, deterministic_queue):
    current_time = 0
    completion_times = [0, 0, 0]
    total_serviced = 0

    if deterministic_queue == 0:
        completion_times[0] += mus[0]

    while current_time < run_time:
        for machine, mu in enumerate(mus):
            if machine != deterministic_queue:
                completion_times[machine] = current_time + sample_exponential(mu)

        return