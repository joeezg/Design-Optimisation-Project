# Simplex Optimisation Algorithm

This repository contains a Python implementation of the Nelder Mead Simplex Algorithm. This iteratively refines a simplex to approximate the minimum of a given equation (function, f).
Applications: for example, determining what components of bernoulli's principle applied to a flow problem would minimise total energy loss

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- Pandas

### Installing 

Clone the repository:

```bash
git clone https://github.com/joeezg/Design-Optimisation-Project.git
cd simplex-optimisation
python simplex_optimisation.py
```

###
This code is designed to fully stop if there's an error within the subfunctions to ensure accuracy in the output

A unit test with 2 successes and 1 failure was conducted on the convergence function

###
Seperate functions were created to separate the methods used for each algorithm step. This improves the clarity of the code despite any redundancies created.
