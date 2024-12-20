from naginterfaces.library import opt
import numpy as np

# Define the covariance matrix
cov_matrix = np.array([
    [0.089618, 0.025347, 0.011072, 0.015178, 0.008784, 0.028007],
    [0.025347, 0.030405, 0.029316, 0.017300, 0.015910, 0.029440],
    [0.011072, 0.029316, 0.051999, 0.015003, 0.019991, 0.029051],
    [0.015178, 0.017300, 0.015003, 0.033967, 0.016893, 0.018864],
    [0.008784, 0.015910, 0.019991, 0.016893, 0.022112, 0.018360],
    [0.028007, 0.029440, 0.029051, 0.018864, 0.018360, 0.036468]
])

# Target active weights and initial guess
target_sum = 0.5
initial_guess = np.full(cov_matrix.shape[0], target_sum / cov_matrix.shape[0])

# Define the objective function
def objective(n, x, needf, f, needg, g):
    # Calculate risk contributions
    marginal_risk = np.dot(cov_matrix, x)
    total_risk = np.sqrt(np.dot(x, marginal_risk))
    contributions = (marginal_risk * x) / total_risk
    # Variance of contributions
    f[:] = np.var(contributions)

    # Gradient (optional)
    if needg:
        n = len(x)
        g_temp = np.zeros(n)
        for i in range(n):
            for j in range(n):
                g_temp[i] += (
                    2 * (contributions[i] - np.mean(contributions)) *
                    (marginal_risk[j] * x[j] / total_risk)
                )
        g[:] = g_temp

# Constraint: Weights sum to target
def sum_constraint(n, x, c, needc, needjac):
    if needc:
        c[:] = [np.sum(x) - target_sum]  # Constraint: Sum of weights = 0.5
    if needjac:
        # Jacobian of sum constraint: [1, 1, 1, ...]
        for i in range(n):
            needjac[0, i] = 1.0

# Lower and upper bounds for weights
lower_bounds = np.zeros(len(initial_guess))  # Non-negative weights
upper_bounds = np.ones(len(initial_guess))  # Example: max weight = 1

# Solve using nlp1_solve
results = opt.nlp1_solve(
    n=len(initial_guess),  # Number of variables
    objfun=objective,  # Objective function
    x0=initial_guess,  # Initial guess
    a=lower_bounds,    # Lower bounds
    b=upper_bounds,    # Upper bounds
    m=1,               # Number of constraints
    cfun=sum_constraint,  # Constraint function
    h=1e-8,  # Tolerance for convergence
)

# Display results
optimal_weights = results.x
print("Optimal Weights:", optimal_weights)
print("Objective Value (Variance):", results.objval)