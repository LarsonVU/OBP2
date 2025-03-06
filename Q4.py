import Q2

# Number of simulations
num_simulations = 100
results = []
max_runtime = 10000
warmup_length = 1000
# Run the simulations
for _ in range(num_simulations):
    first_machine = Q2.run_sim_exponential([1, 1.1, 0.9], [5, 5], max_runtime)
    results.append(first_machine.completed_items / max_runtime)

# Sort results for percentile calculation
results.sort()

# Compute the 95% confidence interval
lower_idx = int(0.025 * num_simulations)
upper_idx = int(0.975 * num_simulations)

# Print only the confidence interval
print(f"[{results[lower_idx]:.4f}, {results[upper_idx]:.4f}]")
print(sum(results)/len(results))