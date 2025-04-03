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
    if num_components < num_required or num_repairmen == 0 or num_components == 0:
        up_time = 0
    else:
        state_matrix = balance_equations(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required)
        up_time = sum([state_matrix[i] for i in range(0, num_components - num_required + 1)])
    return up_time

def compute_cost(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required, component_cost, repair_cost, down_time_cost):
        up_time = compute_up_time(num_components, num_repairmen, failure_rate, repair_rate, warm_standby, num_required)
        return  (num_components * component_cost) + (num_repairmen * repair_cost) + ((1 - up_time) * down_time_cost)



failure_rate_b = 1
repair_rate_b = 1
warm_standby_b = "Yes"
num_required_b = 20

component_cost = 10
repair_cost = 10
down_time_cost = 1000

min_cost = float('inf')
optimal_config = None

for comp in range(1, 170):
    for rep in range(1, 170): 
        cost = compute_cost(comp, rep, failure_rate_b, repair_rate_b, warm_standby_b, num_required_b, component_cost, repair_cost, down_time_cost)
        if cost < min_cost:
            min_cost = cost
            optimal_config = (comp, rep)
            print(f"Current Optimal Configuration: {optimal_config[0]} components and {optimal_config[1]} repairmen with cost {min_cost}")
        elif comp > optimal_config[0] and rep > optimal_config[1] and optimal_config[0] > num_required_b:
            print("break")
            break
    else:
        # Continue outer loop if inner loop wasn't broken
        continue
    # Break outer loop if inner loop was broken
    break

print(f" Optimal Configuration: {optimal_config[0]} components and {optimal_config[1]} repairmen")