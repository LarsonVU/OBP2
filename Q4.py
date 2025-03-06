import Q1
import Q2
import Q3
import numpy as np
import matplotlib.pyplot as plt

### Question 1

def plot_stationary_distribution(pi, access, B1, B2):
    """
    Plots a heatmap of the stationary distribution.

    Parameters:
    - pi: Dictionary containing stationary probabilities.
    - access: Dictionary mapping (i, j, 0, 0) to indices in pi.
    - B1: Integer defining the range of the first dimension.
    - B2: Integer defining the range of the second dimension.
    """
    # Define grid size
    grid_x = B1 + 2
    grid_y = B2 + 2

    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((grid_x, grid_y))

    # Fill the heatmap with stationary distribution values
    for i in range(grid_x):
        for j in range(grid_y):
            heatmap_data[i, j] = pi[access[(i, j, 0, 0)][0]]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap="viridis", origin="lower")

    # Add colorbar
    plt.colorbar(label="Stationary Probability")

    # Label axes
    plt.xlabel("B1")
    plt.ylabel("B2")
    plt.title("Heatmap of Stationary Distribution")

    # Show the plot
    plt.show()

mu1 = 1
mu2 = 1.1
mu3 = 0.9

B1 = 5
B2 = 5 


pi, access = Q1.iterate_prob_matrix([mu1, mu2, mu3], [B1, B2])
throughput_a =  mu1 * (1- sum([pi[access[(B1+1,i,0,0)][0]] for i in range(B2 +2)]))
print("Throughput A:", throughput_a)
plot_stationary_distribution(pi, access, B1, B2)


### Question 2

# Number of simulations
num_simulations = 1000
results = []
max_runtime = 10000
warmup_length = 1000
# Run the simulations
for _ in range(num_simulations):
    first_machine = Q2.run_sim_exponential([1, 1.1, 0.9], [5, 5], max_runtime, warmup_length)
    results.append(first_machine.next.next.completed_items / max_runtime)

# Sort results for percentile calculation
results.sort()

mean = sum(results)/num_simulations
std = (sum([(r-mean)**2 for r in results])/num_simulations)**(0.5)

# Compute the 95% confidence interval (t_999,0.975 = 1.962)
lower = mean - 1.962 * std / num_simulations**(0.5)
upper = mean + 1.962 * std / num_simulations**(0.5)

# Print only the confidence interval
print(f"[{lower:.4f}, {upper:.4f}]")
print("Mean throughput B: ", f"{mean:.4f}")
print("Standard Deviation: ", f"{std:.4f}")

def plot_simulation_results(results):
    """
    Plots the throughput results from multiple simulations.

    Parameters:
    - results: List of throughput values from simulations.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(results, bins=30, color="skyblue")

    # Labels and title
    plt.ylabel("Simulation Run")
    plt.xlabel("Throughput (Completed Items per Unit Time)")
    plt.title("Histogram of simulated throughput")
    
    # Show grid for readability
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show the plot
    plt.show()

#plot_simulation_results(results)

### Question 3
# Number of simulations
num_simulations = 1000
max_runtime = 10000
warmup_length = 1000
# Run the simulations
for i in range(1,4):
    results = []
    print("Deterministic Machine: ", i)
    for _ in range(num_simulations):
        first_machine = Q3.run_sim_deterministic([1, 1.1, 0.9], [5, 5], max_runtime, warmup_length, deterministic_index=i)
        results.append(first_machine.next.next.completed_items / max_runtime)

    # Sort results for percentile calculation
    results.sort()

    mean = sum(results)/num_simulations
    std = (sum([(r-mean)**2 for r in results])/num_simulations)**(0.5)

    # Compute the 95% confidence interval (t_999,0.975 = 1.962)
    lower = mean - 1.962 * std / num_simulations**(0.5)
    upper = mean + 1.962 * std / num_simulations**(0.5)

    # Print only the confidence interval
    print(f"[{lower:.4f}, {upper:.4f}]")
    print("Mean throughput C: ", f"{mean:.4f}")
    print("Standard Deviation: ", f"{std:.4f}")

    #plot_simulation_results(results)
