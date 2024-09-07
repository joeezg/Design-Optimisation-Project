import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to evaluate the function values and sort points
def ev(x1, x2, x0, f):
     """
    Evaluate the function values at three points and return the points sorted by function values.
    
    Args:
        x1 (numpy.ndarray): First point in the simplex.
        x2 (numpy.ndarray): Second point in the simplex.
        x0 (numpy.ndarray): Third point in the simplex (usually the initial guess).
        f (function): The objective function to evaluate.
    
    Returns:
        tuple: Points sorted by function values in ascending order.
    """
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
     """
    Perform the reflection operation on the simplex points.
    
    Args:
        xL (numpy.ndarray): The point with the lowest function value.
        xT (numpy.ndarray): The second highest point.
        xH (numpy.ndarray): The point with the highest function value.
        alpha (float): Reflection coefficient.
        n (int): Dimension of the problem space.
    
    Returns:
        tuple: The centroid (xCE) and the reflected point (xR).
    """
    xCE = (1 / n) * (xT + xL)
    xR = (1 + alpha) * xCE - alpha * xH
    return xCE, xR

# Function to perform expansion
def expansion(xR, xCE, xH, f, gamma=2.0):
     """
    Perform the expansion operation on the simplex to further explore a promising direction.
    
    Args:
        xR (numpy.ndarray): Reflected point.
        xCE (numpy.ndarray): Centroid of the simplex.
        xH (numpy.ndarray): Point with the highest function value.
        f (function): The objective function.
        gamma (float, optional): Expansion coefficient (default is 2.0).
    
    Returns:
        tuple: The expanded point (xE) and updated highest point (xH).
    """
    xE = gamma * xR + (1 - gamma) * xCE
    if f(xE) < f(xR):
        xH = xE
    else:
        xH = xR
    return xE, xH

# Function to check convergence
def convergence(xCE, xT, xL, xH, tol, f):
     """
    Check whether the optimization process has converged based on the tolerance.
    
    Args:
        xCE (numpy.ndarray): Centroid of the simplex.
        xT (numpy.ndarray): Second highest point in the simplex.
        xL (numpy.ndarray): Point with the lowest function value.
        xH (numpy.ndarray): Point with the highest function value.
        tol (float): Tolerance for stopping criterion.
        f (function): The objective function.
    
    Returns:
        tuple: The convergence criterion (float) and a boolean indicating if the tolerance was met.
    """
    criterion = (np.sqrt((f(xL) - f(xCE))**2 + (f(xT) - f(xCE))**2 + (f(xH) - f(xCE))**2)) / 3
    check = criterion < tol
    return criterion, check

# Function to perform outside contraction
def OC(xL, xT, xCE, xR, xH, f, beta, p):
     """
    Perform the outside contraction on the simplex points.
    
    Args:
        xL (numpy.ndarray): Point with the lowest function value.
        xT (numpy.ndarray): Second highest point.
        xCE (numpy.ndarray): Centroid of the simplex.
        xR (numpy.ndarray): Reflected point.
        xH (numpy.ndarray): Point with the highest function value.
        f (function): The objective function.
        beta (float): Contraction coefficient.
        p (float): Shrinking coefficient.
    
    Returns:
        tuple: The outside contracted point (xOC), the updated highest point (xH), and the second highest point (xT).
    """
    xOC = xCE + beta * (xR - xCE)
    if f(xOC) < f(xR):
        xH = xOC
    else:  # Shrinking
        xT, xH = shrinking(xL, xT, xH, p)
    return xOC, xH, xT

# Function to perform inside contraction
def IC(xT, beta, xR, xCE, xH, xL, f, p):
    """
    Perform the inside contraction on the simplex points.
    
    Args:
        xT (numpy.ndarray): Second highest point.
        beta (float): Contraction coefficient.
        xR (numpy.ndarray): Reflected point.
        xCE (numpy.ndarray): Centroid of the simplex.
        xH (numpy.ndarray): Point with the highest function value.
        xL (numpy.ndarray): Point with the lowest function value.
        f (function): The objective function.
        p (float): Shrinking coefficient.
    
    Returns:
        tuple: The inside contracted point (xIC), the updated second highest point (xT), and the highest point (xH).
    """
    xIC = xCE - beta * (xR - xCE)
    if f(xIC) < f(xH):
        xH = xIC
    else:
        xT, xH = shrinking(xL, xT, xH, p)
    return xIC, xT, xH

# Function to perform shrinking
def shrinking(xL, xT, xH, p):
      """
    Perform the shrinking operation to reduce the size of the simplex.
    
    Args:
        xL (numpy.ndarray): Point with the lowest function value.
        xT (numpy.ndarray): Second highest point.
        xH (numpy.ndarray): Point with the highest function value.
        p (float): Shrinking coefficient.
    
    Returns:
        tuple: The updated second highest (xT) and highest points (xH) after shrinking.
    """
    xT = xL + p * (xT - xL)
    xH = xL + p * (xH - xL)
    return xT, xH

# Main Function
def simplex_optimization(f, x0, tol=0.001, alpha=1.0, beta=0.5, gamma=2.0, p=0.5, c = 2.0, n = 2.0):
     """
    Perform the Nelder-Mead Simplex optimization on a given objective function.
    
    Args:
        f (function): The objective function to minimize.
        x0 (numpy.ndarray): Initial guess for the minimum.
        tol (float, optional): Tolerance for convergence. Defaults to 0.001.
        alpha (float, optional): Reflection coefficient. Defaults to 1.0.
        beta (float, optional): Contraction coefficient. Defaults to 0.5.
        gamma (float, optional): Expansion coefficient. Defaults to 2.0.
        p (float, optional): Shrinking coefficient. Defaults to 0.5.
        c (float, optional): Step size. Defaults to 2
        n (float, optional): number of design variables. Defaults to 2
    
    Returns:
        None
    """

    # Define b and a
    b = (c / (n * np.sqrt(2))) * (np.sqrt(n + 1) - 1)
    a = b + c / np.sqrt(2)

    # Initialize points
    x1 = np.array([x0[0] + a, x0[1] + b])
    x2 = np.array([x0[0] + b, x0[1] + a])

    # Evaluate initial points
    xL, xH, xT = ev(x1, x2, x0, f)
    xCE, xR = reflection(xL, xT, xH, alpha, n)
    criterion, check = convergence(xCE, xT, xL, xH, tol, f)
    j = 1

    # Set up plotting
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

        if f(xL) < f(xR) and f(xR) < f(xT):
            xH = xR
            operation = 'reflection'
        elif f(xR) < f(xL):
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

    print('Minimum found at:')
    print(xL)
    print('Minimum function value:')
    print(f(xL))

    # Save data to CSV
    simplex_df = pd.DataFrame(simplex_iterations, columns=['x1', 'x2', 'f(x)', 'Iteration', 'Operation'])
    simplex_df.to_csv('simplex_iterations_final3.csv', index=False)

    plt.show()


if __name__ == "__main__":
    f = lambda x: 5.2 * x[0]**2 + 1.8 * x[1]**2 - 2 * x[0] * x[1] - x[0] - 2 * x[1]
    x0 = np.array([7.0, 5.0])
    simplex_optimization(f, x0)
