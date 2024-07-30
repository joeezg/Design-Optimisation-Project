import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to evaluate the function values and sort points
def ev(x1, x2, x0, f):
    fx = np.array([f(x0), f(x1), f(x2)])
    sorted_indices = np.argsort(fx)
    if sorted_indices[0] == 0:
        return x0, x1, x2
    elif sorted_indices[1] == 1:
        return x1, x0, x2
    else:
        return x2, x1, x0

# Function to perform reflection
def reflection(xL, xT, xH, alpha, n):
    xCE = (1 / n) * (xT + xL)
    xR = (1 + alpha) * xCE - alpha * xH
    return xCE, xR

# Function to perform expansion
def expansion(xR, xCE, xH, f, gamma=2.0):
    xE = gamma * xR + (1 - gamma) * xCE
    if f(xE) < f(xR):
        xH = xE
    else:
        xH = xR
    return xE, xH

# Function to check convergence
def convergence(xCE, xT, xL, xH, tol, f):
    criterion = (np.sqrt((f(xL) - f(xCE))**2 + (f(xT) - f(xCE))**2 + (f(xH) - f(xCE))**2)) / 3
    check = criterion < tol
    return criterion, check

# Function to perform outside contraction
def OC(xL, xT, xCE, xR, xH, f, beta, p):
    xOC = xCE + beta * (xR - xCE)
    if f(xOC) < f(xR):
        xH = xOC
    else:  # Shrinking
        xT, xH = shrinking(xL, xT, xH, p)
    return xOC, xH, xT

# Function to perform inside contraction
def IC(xT, beta, xR, xCE, xH, xL, f, p):
    xIC = xCE - beta * (xR - xCE)
    if f(xIC) < f(xH):
        xH = xIC
    else:
        xT, xH = shrinking(xL, xT, xH, p)
    return xIC, xT, xH

# Function to perform shrinking
def shrinking(xL, xT, xH, p):
    xT = xL + p * (xT - xL)
    xH = xL + p * (xH - xL)
    return xT, xH

# %% Main script
if __name__ == "__main__":
    # Initialise variables
    c = 2
    n = 2
    x0 = np.array([7.0, 5.0])
    tol = 0.001
    beta = 0.5
    alpha = 1.0
    gamma = 2.0
    p = 0.5

    b = (c / (n * np.sqrt(2))) * (np.sqrt(n + 1) - 1)
    a = b + c / np.sqrt(2)

    x1 = np.array([x0[0] + a, x0[1] + b])
    x2 = np.array([x0[0] + b, x0[1] + a])

    f = lambda x: 5.2 * x[0]**2 + 1.8 * x[1]**2 - 2 * x[0] * x[1] - x[0] - 2 * x[1]

    # Evaluate initial points
    xL, xH, xT = ev(x1, x2, x0, f)
    xCE, xR = reflection(xL, xT, xH, alpha, n)
    criterion, check = convergence(xCE, xT, xL, xH, tol, f)
    j = 1

    # Set up the figure for plotting
    plt.figure()
    plt.axis('equal')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Simplex Changes During Iterations')

    simplex_iterations = []

    operation = 'Initial simplex'
    while not check:
        # Plot the simplex
        simplex = np.array([xL, xT, xH, xL]).T
        plt.plot(simplex[0, :], simplex[1, :], '-x', label=f'Iteration {j}')
        plt.legend()

        # Evaluate and reflect
        xL, xH, xT = ev(xL, xH, xT, f)
        xCE, xR = reflection(xL, xT, xH, alpha, n)

        simplex_iterations.extend([
            (xL[0], xL[1], f(xL), j, operation),
            (xT[0], xT[1], f(xT), j, operation),
            (xH[0], xH[1], f(xH), j, operation)
        ])

        # Perform the appropriate operation based on the function values
        if f(xL) < f(xR) and f(xR) < f(xT):  # 1
            xH = xR
            operation = 'reflection'
        elif f(xR) < f(xL):  # 2
            xE, xH = expansion(xR, xCE, xH, f)
            operation = 'expansion'
        elif f(xT) < f(xR) and f(xR) < f(xH):
            xOC, xH, xT = OC(xL, xT, xCE, xR, xH, f, beta, p)
            operation = 'Outside Contraction'
        elif f(xR) > f(xH):
            xIC, xT, xH = IC(xT, beta, xR, xCE, xH, xL, f, p)
            operation = 'Inside Contraction'

        criterion, check = convergence(xCE, xT, xL, xH, tol, f)
        j += 1

    j -= 1

    print('Minimum found at:')
    print(xL)
    print('Minimum function value:')
    print(f(xL))

    # Convert list to DataFrame and export to CSV
    simplex_df = pd.DataFrame(simplex_iterations, columns=['x1', 'x2', 'f(x)', 'Iteration', 'Operation'])
    simplex_df.to_csv('simplex_iterations_final3.csv', index=False)

    plt.show()
