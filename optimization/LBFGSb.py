from scipy.optimize import minimize

def optimize_with_lbfgsb(func, x0, bounds=None, maxiter=1000):
    result = minimize(func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})
    return result