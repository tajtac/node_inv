import numpy as np
import matplotlib.pyplot as plt

# Define a test function and its analytical derivatives
def test_function(X, Y):
    """More complex test function: exp(X) * sin(pi*Y) + X^2 * Y^3"""
    return np.exp(X) * np.sin(np.pi * Y) + X**2 * Y**3

def analytical_derivatives(X, Y):
    """Analytical derivatives of the more complex test function"""
    dfdx = np.exp(X) * np.sin(np.pi * Y) + 2 * X * Y**3
    dfdy = np.pi * np.exp(X) * np.cos(np.pi * Y) + 3 * X**2 * Y**2
    return dfdx, dfdy
def richardson_derivative(Z, h):
    """Calculates the derivatives using finite difference and Richardson extrapolation for step size h."""
    Z_x = np.zeros_like(Z)
    Z_y = np.zeros_like(Z)
    rows, cols = Z.shape

    # Loop through all points to calculate derivatives
    for i in range(rows):
        for j in range(cols):
            # Derivative in x-direction
            if i == 0:  # Left boundary (use forward difference)
                Z_y[i, j] = (Z[i + 1, j] - Z[i, j]) / h
            elif i == rows - 1:  # Right boundary (use backward difference)
                Z_y[i, j] = (Z[i, j] - Z[i - 1, j]) / h
            else:  # Interior points (use central difference and Richardson)
                # Central difference with step h
                dy_h = (Z[i + 1, j] - Z[i - 1, j]) / (2 * h)
                
                # Approximation for h/2 by averaging values
                f_y_plus_h_half = (Z[i + 1, j] + Z[i, j]) / 2
                f_y_minus_h_half = (Z[i - 1, j] + Z[i, j]) / 2
                
                dy_h_over_2 = (f_y_plus_h_half - f_y_minus_h_half) / h

                # Richardson Extrapolation
                Z_y[i, j] = (4 * dy_h_over_2 - dy_h) / 3

            # Derivative in y-direction
            if j == 0:  # Bottom boundary (use forward difference)
                Z_x[i, j] = (Z[i, j + 1] - Z[i, j]) / h
            elif j == cols - 1:  # Top boundary (use backward difference)
                Z_x[i, j] = (Z[i, j] - Z[i, j - 1]) / h
            else:  # Interior points (use central difference and Richardson)
                # Central difference with step h
                dx_h = (Z[i, j + 1] - Z[i, j - 1]) / (2 * h)
                
                # Approximation for h/2 by averaging values
                f_x_plus_h_half = (Z[i, j + 1] + Z[i, j]) / 2
                f_x_minus_h_half = (Z[i, j - 1] + Z[i, j]) / 2
                
                dx_h_over_2 = (f_x_plus_h_half - f_x_minus_h_half) / h

                # Richardson Extrapolation
                Z_x[i, j] = (4 * dx_h_over_2 - dx_h) / 3

    return Z_x,Z_y

# Function to test Richardson extrapolation
def test_richardson_extrapolation(grid_size, h):
    """Tests the richardson_derivative function using a known test function."""
    # Generate a regular grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute the function values at the grid points
    Z = test_function(X, Y)

    # Calculate the analytical derivatives for comparison
    dfdx_analytical, dfdy_analytical = analytical_derivatives(X, Y)

    # Use Richardson Extrapolation to estimate the derivatives numerically
    Z_x_richardson, Z_y_richardson = richardson_derivative(Z, h)

    # Compare the numerical and analytical derivatives
    error_x = np.abs(Z_x_richardson - dfdx_analytical)
    error_y = np.abs(Z_y_richardson - dfdy_analytical)

    # Print maximum error to evaluate accuracy
    print("Maximum error in x-derivative:", np.max(error_x))
    print("Maximum error in y-derivative:", np.max(error_y))

    # Plot the numerical and analytical derivatives and their difference
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.contourf(X, Y, Z_x_richardson, 20, cmap='viridis')
    plt.title('Numerical dfdx (Richardson)')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.contourf(X, Y, dfdx_analytical, 20, cmap='viridis')
    plt.title('Analytical dfdx')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.contourf(X, Y, error_x, 20, cmap='viridis')
    plt.title('Error in dfdx')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.contourf(X, Y, Z_y_richardson, 20, cmap='viridis')
    plt.title('Numerical dfdy (Richardson)')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.contourf(X, Y, dfdy_analytical, 20, cmap='viridis')
    plt.title('Analytical dfdy')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.contourf(X, Y, error_y, 20, cmap='viridis')
    plt.title('Error in dfdy')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Example test using a 50x50 grid and step size h
grid_size = 100
h = 1.0 / (grid_size - 1)

# Assuming richardson_derivative is already implemented
test_richardson_extrapolation(grid_size, h)