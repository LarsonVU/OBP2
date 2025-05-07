# def compute_up_time_gurobi(model, failure_rate, repair_rate, warm_standby, num_required, num_components, num_repairmen, max_components, max_repairmen):
#     lambda_mu_ratio = failure_rate / repair_rate
    
#     # Auxiliary variables
#     pi_0 = model.addVar(vtype=GRB.CONTINUOUS, name="pi_0")
#     state_probs = [model.addVar(vtype=GRB.CONTINUOUS, name=f"pi_{j}") for j in range(num_components + 1)]
    
#     # Constraints for probability normalization
#     if warm_standby == "Yes":
#         sum1_expr = quicksum(math.comb(num_components, j) * (lambda_mu_ratio ** j) for j in range(0, num_repairmen + 1))
#         sum2_expr = quicksum((math.factorial(num_components) / (math.factorial(num_components - j) * math.factorial(num_repairmen) * (num_repairmen ** (j - num_repairmen)))) * (lambda_mu_ratio ** j)
#                               for j in range(num_repairmen + 1, num_components + 1))
#         model.addConstr(pi_0 == 1 / (sum1_expr + sum2_expr), name="pi_0_constraint")
        
#         for j in range(0, num_components + 1):
#             if j <= num_repairmen:
#                 model.addConstr(state_probs[j] == pi_0 * (math.comb(num_components, j) * (lambda_mu_ratio ** j)), name=f"pi_{j}_constraint")
#             else:
#                 model.addConstr(state_probs[j] == pi_0 * ((math.factorial(num_components) / (math.factorial(num_components - j) * math.factorial(num_repairmen) * (num_repairmen ** (j - num_repairmen)))) * (lambda_mu_ratio ** j)),
#                                 name=f"pi_{j}_constraint")
#     else:
#         def pi_0_inverse_expr():
#             if num_repairmen <= num_components - num_required + 1:
#                 sum1 = quicksum((num_required ** j) / math.factorial(j) * (lambda_mu_ratio ** j) for j in range(0, num_repairmen + 1))
#                 sum2 = quicksum((num_required ** j) / (math.factorial(num_repairmen) * (num_repairmen ** (j - num_repairmen))) * (lambda_mu_ratio ** j) for j in range(num_repairmen + 1, num_components - num_required + 2))
#                 return sum1 + sum2
#             else:
#                 return quicksum((num_required ** j) / math.factorial(j) * (lambda_mu_ratio ** j) for j in range(0, num_components - num_required + 2))
        
#         model.addConstr(pi_0 == 1 / pi_0_inverse_expr(), name="pi_0_constraint")
        
#         for j in range(0, num_components - num_required + 2):
#             if j <= num_repairmen:
#                 model.addConstr(state_probs[j] == pi_0 * ((num_required ** j) / math.factorial(j) * (lambda_mu_ratio ** j)), name=f"pi_{j}_constraint")
#             else:
#                 model.addConstr(state_probs[j] == pi_0 * ((num_required ** j) / (math.factorial(num_repairmen) * (num_repairmen ** (j - num_repairmen))) * (lambda_mu_ratio ** j)), name=f"pi_{j}_constraint")
    
#     # Compute up-time probability
#     up_time = quicksum(state_probs[i] for i in range(0, num_components - num_required + 1))
#     return up_time

# def optimize_system(failure_rate, repair_rate, warm_standby, num_required, component_cost, repair_cost, down_time_cost, max_components, max_repairmen):
#     model = Model("System Optimization")
    
#     # Decision variables
#     num_components = model.addVar(vtype=GRB.INTEGER, name="num_components", lb=num_required, ub=max_components)
#     num_repairmen = model.addVar(vtype=GRB.INTEGER, name="num_repairmen", lb=1, ub=max_repairmen)
#     can_operate  = model.addVar(vtype=GRB.BINARY, name="can_operate")
#     up_time_var = model.addVar(vtype=GRB.CONTINUOUS, name="up_time", lb=0, ub=1)
    
#     # Compute up-time using Gurobi-integrated function
#     up_time = compute_up_time_gurobi(model, failure_rate, repair_rate, warm_standby, num_required, num_components, num_repairmen, max_components, max_repairmen)
    
#     model.addConstr(up_time_var <= can_operate * up_time, "up_time_constraint")
#     model.addConstr(num_components >= num_required * can_operate, "num_components_constraint")
    
#     # Objective function: minimize total cost
#     model.setObjective(
#         (num_components * component_cost) + (num_repairmen * repair_cost) + ((1 - up_time_var) * down_time_cost),
#         GRB.MINIMIZE)
    
#     # Optimize
#     model.optimize()
    
#     # Return optimized values
#     return {
#         "num_components": num_components.X,
#         "num_repairmen": num_repairmen.X,
#         "total_cost": model.ObjVal
#     }
