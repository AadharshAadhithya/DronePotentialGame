import numpy as np
import sys
import os

# Add parent directory to path to import Player
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player import Player


def test_player():
    """Test that Player.cost returns a scalar value."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define dimensions
    d = 5  # state dimension
    m = 3  # control dimension
    tau = 4  # number of time steps
    
    # Initialize Q to be random dxd matrix
    Q = np.random.rand(d, d)
    # Make Q positive semi-definite (Q = Q^T @ Q)
    Q = Q.T @ Q
    
    # Initialize R to be random mxm matrix
    R = np.random.rand(m, m)
    # Make R positive semi-definite (R = R^T @ R)
    R = R.T @ R
    
    # Create random x in R^d for tau durations (shape: d x tau)
    x = np.random.rand(d, tau)
    
    # Create random u in R^m for tau durations (shape: m x tau)
    u = np.random.rand(m, tau)
    
    # Create reference trajectory (same shape as x)
    xref = np.random.rand(d, tau)
    
    # Create a dummy dynamics function f (not used in cost, but required for Player)
    def f(xt, ut):
        return xt + ut[:d] if ut.shape[0] >= d else xt
    
    # Create a dummy constraint function g (not used in cost, but required for Player)
    def g(xt, ut):
        return np.array([0.0])
    
    # Create Player instance
    player = Player(xref=xref, f=f, g=g, tau=tau, Qi=Q, Qtau=Q, Ri=R)
    
    # Call cost method
    cost_value = player.cost(x, u)
    
    # Check that cost returns a scalar
    assert np.isscalar(cost_value) or (isinstance(cost_value, np.ndarray) and cost_value.ndim == 0), \
        f"cost should return a scalar, but got shape {np.shape(cost_value)}"
    
    # Convert to Python scalar if it's a 0-d array
    cost_scalar = float(cost_value)
    
    print(f"✓ Test passed: cost returned a scalar value: {cost_scalar}")
    print(f"  Dimensions: d={d}, m={m}, tau={tau}")
    print(f"  Q shape: {Q.shape}, R shape: {R.shape}")
    print(f"  x shape: {x.shape}, u shape: {u.shape}")
    
    return cost_scalar


if __name__ == "__main__":
    test_player()

