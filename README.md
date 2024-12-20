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
def objective(weights):
    # Calculate risk contributions
    marginal_risk = np.dot(cov_matrix, weights)
    total_risk = np.sqrt(np.dot(weights, marginal_risk))
    contributions = (marginal_risk * weights) / total_risk
    # Return variance of contributions
    return np.var(contributions)

# Define the gradient of the objective function
def gradient(weights):
    marginal_risk = np.dot(cov_matrix, weights)
    total_risk = np.sqrt(np.dot(weights, marginal_risk))
    contributions = (marginal_risk * weights) / total_risk

    # Gradient calculation
    n = len(weights)
    grad = np.zeros(n)
    for i in range(n):
        for j in range(n):
            grad[i] += (
                2 * (contributions[i] - np.mean(contributions)) *
                (marginal_risk[j] * weights[j] / total_risk)
            )
    return grad

# Constraint: Weights sum to target
def constraint_sum(weights):
    return np.sum(weights) - target_sum

# Constraint: Weights must be non-negative
def constraint_non_negative(weights):
    return weights

# Setup bounds (non-negative weights)
lower_bounds = np.zeros(len(initial_guess))
upper_bounds = np.ones(len(initial_guess))  # Example: max weight = 1

# Solve the optimization problem
results = opt.nlp_solve(
    objfun=objective,
    x0=initial_guess,
    objgrd=gradient,
    n=initial_guess.size,
    bl=lower_bounds,
    bu=upper_bounds,
    cns=[(constraint_sum, 0)],  # Sum-to-target constraint
)

# Display results
optimal_weights = results.x
print("Optimal Weights:", optimal_weights)
print("Objective Value (Variance):", results.objval)
