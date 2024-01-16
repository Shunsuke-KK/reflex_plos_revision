# Overview

This page contains files related to the paper, "Identifying essential factors for energyefficient walking control across a wide range of velocities in reflex-based musculoskeletal systems" 
(https://doi.org/10.1371/journal.pcbi.1011771), by S Koseki, M Hayashibe, and D Owaki.

## folder
*  `assets`: musculoskeletal models.
*  `cmaes`: cmaes (optimizing algorithm) codes. I used (https://pypi.org/project/cmaes/), and some functions were added for this study.
*  `custom_env`: gym-like environment for musculoskeletal model is contained. The environments include reset function, step function, fall down detect function, sensory function, and so on.
*  `func_data`: polynimial functions of control parameters.
*  `graph`: graphs and their codes in the paper are here.
*  `pic`: snapshots for different target velocities.
*  `reflex_opt`: main codes for optimizing parameters and running the results.

## file
*  `optimize.py`: collect dataset using the optimization (CMAES) algorithm.
*  `optimize_costfunc.py`: collect dataset undeer the different weight coefficients in the objective cost function (different cost function).
*  `optimize_shortleg.py`: collect dataset using the different bipedal model, where its segment lengths are shortened by 20% from the original (different body structure).
*  `optimize_timedelay.py`: collect dataset with doubling the time delay of sensory information transmission to the controller (different neural system).
*  `run.py`: run the simulation result using dataset obtained from the `optimize.py`.
*  `run_costfunc.py`: run the simulation result using dataset obtained from the `optimize_costfunc.py`.
*  `run_shortleg.py`: run the simulation result using dataset obtained from the `optimize_shortleg.py`.
*  `run_timedelay.py`: run the simulation result using dataset obtained from the `optimize_timedelay.py`.

# Important Package Versions for Troubleshooting Errors
*  **MuJoCo**: mujoco200
*  **mujoco-py**: 2.0.2.13
*  **pandas**: 1.4.2
*  **matplotlib**: 3.5.2

# Optimizing Control Parameters
1. run `optimize.py`

# Performence-Weighted Least Square (PWLS) method
You can find the Python implementation of the PWLS below (those who want to use PWLS shold use this, because codes in the paper maybe messy).
```
def PWLS(x, y, beta, degree):
    """
    This function calculates the coefficients of polynomial that minimizes the weighted square error.
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
    w = np.dot(np.dot(np.linalg.inv(np.dot((B*X).T, (B*X))), (B*X).T), beta*y)
    w = w.reshape(-1)
    return w
```
then, you can depict the graph using the caluculated coefficients like this 
```
def f(x,w):
    y = 0
    for i, w in enumerate(w):
        y += w * (x ** i)
    return y

w_PWLS = PWLS(x, y, beta, 6)
xlines = np.linspace(0, 10, 200)
plt.plot(xlines, f(xlines,w_PWLS))
```

Here, I provide detailed implementations using a simple example.

we prepared dataset with weight (=beta) for each data point as follows;
```
x    = np.array([1,  1, 1, 2, 2,  2, 3, 3, 3,     4, 4, 4,  5, 5, 5,  6, 6, 6,    7, 7, 7, 8,  8, 8, 9, 9,  9])
y    = np.array([1,  2, 3, 1, 2,  3, 1, 2, 3,     7, 8, 9,  7, 8, 9,  7, 8, 9,    4, 5, 6, 4,  5, 6, 4, 5,  6])
beta = np.array([10, 1, 1, 1, 10, 1, 1, 1, 10,   10, 1, 1, 10, 1, 1, 10, 1, 1,   10, 1, 1, 1, 10, 1, 1, 1, 10])
```
where larger beta value indicate the corresponding data point is highly-evaluated.

We can see this dataset visually through the graph as follows;
![pwls_draft](https://github.com/Shunsuke-KK/reflex_plos_revision/assets/78842615/788c4074-7186-4bfa-af73-117fd8dd16ea)

The following graph compares the calculated polynomials between the normal least square and PWLS. We can find that the calculated polynomial through PWLS passes close to higher-evaluated data points!!

```
def NormalLeastSquare(x, y, degree):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X = np.empty((len(x),0))
    for i in range(0, degree+1):
        current_x = x ** i
        X = np.hstack((X, current_x))
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    w = w.reshape(-1)
    return w

w_PWLS = PWLS(x, y, beta, 6)
w_NormalLS = NormalLeastSquare(x, y, 6)

plt.plot(xlines, f(xlines,w_NormalLS), label='Normal Least Square')
plt.plot(xlines, f(xlines,w_PWLS), label='PWLS')
```

![comparison](https://github.com/Shunsuke-KK/reflex_plos_revision/assets/78842615/48916aea-f38d-4477-8c9a-1ca87251ceb5)

