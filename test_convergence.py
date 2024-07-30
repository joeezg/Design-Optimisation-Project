import unittest
import numpy as np
from simplex_optimisation import convergence

class TestConvergence(unittest.TestCase):

    def test_successful_scenario_1(self):
        # Convergence should succeed with adjusted points
        f = lambda x: x**2
        xCE = np.array([1])
        xT = np.array([1.01])
        xL = np.array([0.99])
        xH = np.array([1.02])
        tol = 0.1
        
        criterion, check = convergence(xCE, xT, xL, xH, tol, f)
        self.assertLess(criterion, tol)
        self.assertTrue(check)

    def test_successful_scenario_2(self):
        # Convergence should succeed
        f = lambda x: np.sin(x)
        xCE = np.array([np.pi/4])
        xT = np.array([np.pi/4 + 0.01])
        xL = np.array([np.pi/4 - 0.01])
        xH = np.array([np.pi/4 + 0.005])
        tol = 0.01
        
        criterion, check = convergence(xCE, xT, xL, xH, tol, f)
        self.assertLess(criterion, tol)
        self.assertTrue(check)

    def test_failure_scenario(self):
        # Convergence should fail
        f = lambda x: x**3
        xCE = np.array([1])
        xT = np.array([2])
        xL = np.array([0])
        xH = np.array([3])
        tol = 0.5
        
        criterion, check = convergence(xCE, xT, xL, xH, tol, f)
        self.assertGreaterEqual(criterion, tol)
        self.assertFalse(check)

if __name__ == "__main__":
    unittest.main()
