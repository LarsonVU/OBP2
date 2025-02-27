import Q2

# Run the simulation with the given parameters
for i in range(10):
    first_machine = Q2.run_sim_exponential([1, 1.1, 0.9], [5, 5], 100000)
    print(first_machine.completed_items / 100000)