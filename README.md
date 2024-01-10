# Overview

This page contains files related to the paper, "Identifying essential factors for energyefficient walking control across a wide range of velocities in reflex-based musculoskeletal systems" 
(https://doi.org/10.1371/journal.pcbi.1011771).

*  **assets**: musculoskeletal models.
*  **cmaes**: cmaes (optimizing algorithm) codes. I used (https://pypi.org/project/cmaes/), and some functions were added for this study.
*  **custom_env**: gym-like environment for musculoskeletal model is contained. The environments include reset function, step function, fall down detect function, sensory function, and so on.
*  **func_data**: polynimial functions of control parameters.
*  **graph**: graphs and their codes in the paper are here.
*  **pic**: snapshots for different target velocities.
*  **reflex_opt**: main codes for optimizing parameters and running the results.

# Important Package Versions for Troubleshooting Errors
**MuJoCo**: mujoco200
**mujoco-py**: 2.0.2.13
**pandas**: 1.4.2
matplotlib: 3.5.2

# Optimizing Control Parameters

# Performence-Weighted Least Square (PWLS)
You can find the Python implementation of the PWLS below.
```
def PWLS(x, y, beta, degree):
    """
    x, y (NumPy array): These represent the data points. Please provide 1-dimensional NumPy arrays.
        e.g., x = np.array([0.0, 1.0, 2.5]), y = np.array([1.0, 0.5, 3.5])

    beta (NumPy array): Weights corresponding to each data point. Please provide a 1-dimensional NumPy array of the same length as x and y.
        e.g., beta = np.array([1.0, 1.0, 2.0])

    degree (int): This denotes the degree of the polynomial function. For example, setting degree=2 will calculate a quadratic curve.
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    beta = beta.reshape(-1, 1)

    B = np.hstack([beta]*(degree+1))
    X = np.empty((len(x),0))
    for i in range(0, degree+1):
        current_x = x ** i
        X = np.hstack((X, current_x))
    # w_0*(x**0) + w_1*(x**1) + ... + w_m*(x**m)
    w = np.dot(np.dot(np.linalg.inv(np.dot((B*X).T, (B*X))), (B*X).T), beta*y)

    def f(x):
        y = 0
        for i, w in enumerate(w):
            y += w * (x ** i)
        return y

    return f, w
```
