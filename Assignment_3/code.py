import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, PulpSolverError
import math


def load_data(filename="actuals.csv"):
    df = pd.read_csv(filename, header=None)
    df.columns = ["day", "volume"]
    df = df.dropna()
    df = df.astype({"day": int, "volume": float})
    return df

def compute_wape(test_fit, test_actuals):
    wape = np.sum(np.abs(test_fit - test_actuals)) / np.sum(test_actuals)
    return wape

def polynomial_forecast(df, degree=3, forecast_days=7):
    df["day_of_week"] = df["day"] % 7  # Intra-week seasonality
    df["day_of_year"] = df["day"] % 365  # Intra-year seasonality
    df = pd.get_dummies(df, columns=["day_of_week", "day_of_year"], drop_first=True)
    train = df[:-70]
    test = df[-70:]

    poly = PolynomialFeatures(degree=degree)

    X_train = poly.fit_transform(train[["day"]])
    X_train = np.hstack([X_train, train.drop(columns=["day", "volume"]).values])
    y_train = train["volume"]
    model = LinearRegression().fit(X_train, y_train)

    X_test = poly.transform(test[["day"]])
    X_test = np.hstack([X_test, test.drop(columns=["day", "volume"]).values])
    y_pred = model.predict(X_test)

    wape = compute_wape(y_pred, test["volume"].values)
    fit = model.predict(np.hstack([poly.fit_transform(df[["day"]]), 
                                    df.drop(columns=["day", "volume"]).values]))

    # Forecast week 260
    forecast_days_idx = np.arange(1456, 1456 + forecast_days+1).reshape(-1, 1)
    weekdays = np.tile(np.arange(7), forecast_days +1 // 7 + 1)[:forecast_days+1]
    year_days = np.tile(np.arange(365), forecast_days +1 // 365 + 1)[:forecast_days +1]
    forecast_df = pd.DataFrame({"day_of_week": weekdays, "day_of_year": year_days})
    print(forecast_df.shape)
    forecast_df = pd.get_dummies(forecast_df, columns=["day_of_week", "day_of_year"], drop_first=True)
    X_forecast = np.hstack([poly.transform(forecast_days_idx), forecast_df.values])
    forecast = model.predict(X_forecast)

    # Print coefficients for weekday and polynomial trend
    coefficients = model.coef_
    weekday_coefficients = coefficients[poly.n_output_features_:poly.n_output_features_+7] 
    polynomial_coefficients = coefficients[:poly.n_output_features_]  # Polynomial coefficients

    print("Polynomial Coefficients:")
    for i, coef in enumerate(polynomial_coefficients):
        print(f"Degree {i}: {coef:.4f}")

    print("\nWeekday Coefficients:")
    for i, coef in enumerate(weekday_coefficients):
        print(f"Weekday {i + 1}: {coef:.4f}")

    return wape, fit, forecast


def plot_forecast(actuals, fit, forecast):
    # Plot the actual data
    plt.figure(figsize=(12, 6))
    plt.scatter(actuals["day"], actuals["volume"], label="Actual Data", color="blue", s=10)

    # Plot the polynomial fit
    days = np.arange(actuals["day"].min(), actuals["day"].max() + 1).reshape(-1, 1)
    plt.plot(days, fit, label="Polynomial Fit", color="red")

    # Plot the forecast
    forecast_days = np.arange(actuals["day"].max() + 1, actuals["day"].max() + 1 + len(forecast))
    plt.plot(forecast_days, forecast, label="Forecast", color="green")

    # Add labels and legend
    plt.xlabel("Day")
    plt.ylabel("Volume")
    plt.title("Polynomial Fit and Forecast")
    plt.legend()
    plt.grid()
    plt.show()

def erlang_a_model(service_level_0, volume, aht, patience, opening_hours):
    R = volume * aht
    gamma = 1 / patience
    
    def delay_prob(c):
        probs = [1]
        for k in range(1, c):
            probs.append((R ** k) / math.factorial(k) *probs[0])
        
        for k in range(c + 1, 200):
            prob = probs[c-1] * np.prod([volume / ((c * aht) + (i - c)*gamma) for i in range(c + 1, k + 1)])
            probs.append(prob * probs[0])
        probs = probs / sum(probs)

        Pdelay = sum(probs[c:])
        return Pdelay

    c = int(np.ceil(R))  # Start from traffic intensity
    while (1-delay_prob(c)) <= service_level_0:
        c += 1

    return opening_hours*c, c


def optimize_shifts_integer(required_hours):
    from itertools import combinations
    import pulp

    shift_types = []

    # Create all valid 3-day 8h shifts (Mon–Fri)
    for start in range(5):  # days 0–4
        shift = [0]*7
        for i in range(3):
            shift[(start + i) % 7] = 8
        shift_types.append(shift)

    # Create all valid 4-day 6h shifts (Mon–Thu)
    for start in range(4):  # days 0–3
        shift = [0]*7
        for i in range(4):
            shift[(start + i) % 7] = 6
        shift_types.append(shift)

    shift_types = np.array(shift_types).T  # Shape: 7 x N_shifts (days × shifts)
    print("Shift types:\n", shift_types)
    num_shifts = shift_types.shape[1]

    # Cost per shift: total hours per shift
    shift_costs = [int(sum(shift_types[:, i])) for i in range(num_shifts)]

    # Define LP problem
    prob = LpProblem("Shift_Optimization", LpMinimize)

    # Create integer decision variables: x0 ... xN (number of times each shift is assigned)
    x = [LpVariable(f"x{i}", lowBound=0, cat=LpInteger) for i in range(num_shifts)]

    # Objective: Minimize total scheduled hours
    prob += lpSum(shift_costs[i] * x[i] for i in range(num_shifts))

    # Constraints: For each day, scheduled hours must meet required_hours
    for day in range(7):
        prob += lpSum(shift_types[day, i] * x[i] for i in range(num_shifts)) >= required_hours[day]

    # Solve
    try:
        prob.solve()
        if prob.status != 1:  # 1 means optimal
            return None, None, None
    except PulpSolverError:
        return None, None, None

    # Get solution
    x_vals = np.array([x[i].varValue for i in range(num_shifts)])
    scheduled = shift_types @ x_vals
    inefficiency = (sum(scheduled) - sum(required_hours)) / sum(required_hours)

    return x_vals, scheduled, inefficiency


data = load_data()
wape, fit, forecast = polynomial_forecast(data, degree=3, forecast_days=52*7)
print(f"WAPE: {wape:.2%}")
print("Forecast for the week 260:", forecast[-7:])
plot_forecast(data, fit, forecast)

volume = np.mean(forecast[-7:]) /14 # daily volume
aht = 5 / 60 #aht in hours
patience = 10 /60 # patience in hours
opening_hours = 14 *7 # 14 hours a day, 7 days a week
sevice_level_0 = 0.60
c, agents = erlang_a_model(sevice_level_0, volume, aht, patience, opening_hours)
print(f"Number of agents-hours needed: {c:.2f}")

c_quarters = []
quarters = np.linspace(0, 2, 7)
for i in quarters:
    c_quarter, agents = erlang_a_model(sevice_level_0, volume * i, aht, patience, opening_hours)
    c_quarters.append(c_quarter)	

needed_agents = np.mean(c_quarters)
print(f"Number of agents-hours needed (quarters): {needed_agents:.2f}")

result, scheduled, inefficiency = optimize_shifts_integer(c_quarters)

print("Optimal shifts:\n", [int(result[i]) for i in range(len(result))])
print("Scheduled hours:\n", scheduled)
print("Required hours:\n", c_quarters)
print("Inefficiency:\n", inefficiency)


