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
*  `optimization.depelope.py`: make the the graph of the optimization convergence curve, which is shown in the supplemantary material.

# Important Package Versions for Troubleshooting Errors
*  **MuJoCo**: mujoco200
*  **mujoco-py**: 2.0.2.13
*  **pandas**: 1.4.2
*  **matplotlib**: 3.5.2

# Optimizing Control Parameters
1. run `optimize.py`
2. you would see
    ```
    select 1 or 2:
    1 (incrementary increase)
    2 (incrementary decrease)
    >>
    ```
    In `1`, initially, the target velocity, $`v^{tar}_{x}`$, in cost function $`f`$ (Eq (12)) is set to 1.3 m/s, which is the human selfselected speed. 
    Control parameter set, that generate a gait around $`v^{tar}_{x}`$ are obtained.
    Then, $`v^{tar}_{x}`$ is slightly increased to $`v^{tar}_{x}+\Delta v_{x}`$, and the corresponding control parameter set around the updated target velocity are collected.
    This process is repeated until $`v^{tar}_{x}`$ reaches the upper limit of the target velocity, $`v^{tar}_{x\,max}`$.

    In `2`, $`v^{tar}_{x}`$ is initially set to 1.2 m/s. 
    Then, $`v^{tar}_{x}`$ is slightly decreased to $`v^{tar}_{x}-\Delta v_{x}`$ and this process is also repeated until $`v^{tar}_{x}`$ reaches the lower limit of the target velocity, $`v^{tar}_{x\,min}`$. 
    The initial control parameters are set identically in both situation, which generates steady walking gait of 1.25 m/s.
3. select `1` or `2` and press Enter
4. an optimization launch, this process is saved to `reflex_opt/save_data/` as a .pickle file. For example, if you select `1` in the above, you would see the file named `reflex_opt/save_data/review_forw_new`, which contains `logger.pickle` and `vel_gen0.pickle`.
5. you can run several programs in parallel to collect the data set efficiently.
6. sometimes, the optimization is forced to terminate with the following message:
   ```
    "WARNING: Nan, Inf or huge value in QACC at DOF xx. The simulation is unstable. Time = <sim time value>"
   ```
   To re-start the optimization from the terminated point, please go to `reflex_opt/opt_forw.py` or `reflex_opt/opt_back.py`.
   Comment out "previous_data = False", which is around line 469, and make "previous_data = True".
   Then, please specify folder name that is stored in "/reflex_opt/save_data/~"


# Run the Bipedal Model
1. run `run.py`
2. you would see
    ```
    Is there a saved pickle data??
    If yes, please type "1"
    >>
    ```
    because calculating optimized functions take a little long time, you can save the coefficients of optimized polynomial functions as described later.
3. 1. if you do not select `1`:
      the coefficients of polynomial are calculated using specified .pickle files, which are generated in `optimized.py`.
      you can chage the dataset for optimization in `run.py`. In original setting, a prepared .pickle file is specified as follows.
      ```
      folder_name1 = 'review_back_1'
      folder_name2 = 'review_forw_1'
      ```
      after the calculation of optimized polynomial functions, you would see
      ```
      save as a pickle data??\nIf yes, please input "1"
      >>
      ```
      if you want to save, please press 1 and name the file after the following message.
      ```
      please name the save pickle file
      >>
      ```
   2. if you select `1`:
      input the .pickle file name after the following message:
      ```
      please type its name without ".pickle"
      (sample: func_cost1_A1000000)
      >>
      ```
4. If all goes fine, the bipeddal model is displayed as follows:
![walking_control2](https://github.com/Shunsuke-KK/reflex_plos_revision/assets/78842615/f0b3f564-1564-4cd5-9ed8-b2aaa982086d)
The walking speed can be controlled by sliding the control bar.
It is not robust enough, please adjust the control bar carefully (reproduce more robust speed transitions is one of the important future work).

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

