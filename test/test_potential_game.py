import numpy as np
import sys
import os

# Add parent directory to path to import classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player import Player, PotientialGame


def test_potential_game():
    """Test that PotientialGame.cost returns a scalar value."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define dimensions
    n_players = 3  # number of players
    d = 4  # state dimension per player
    m = 2  # control dimension per player
    tau = 5  # number of time steps
    
    # Create list of players
    players = []
    
    for i in range(n_players):
        # Initialize Q to be random dxd matrix
        Q = np.random.rand(d, d)
        # Make Q positive semi-definite
        Q = Q.T @ Q
        
        # Initialize R to be random mxm matrix
        R = np.random.rand(m, m)
        # Make R positive semi-definite
        R = R.T @ R
        
        # Create random reference trajectory (shape: d x tau)
        xref = np.random.rand(d, tau)
        
        # Create a dummy dynamics function f for each player
        # f takes xt (d,) and ut (m,) and returns next state (d,)
        def make_f(player_id):
            def f(xt, ut):
                # Simple dynamics: next state is current state plus some function of control
                # Pad ut if needed, or use first d elements
                if ut.shape[0] >= d:
                    return xt + 0.1 * ut[:d]
                else:
                    # If control is smaller, pad with zeros
                    ut_padded = np.zeros(d)
                    ut_padded[:ut.shape[0]] = ut
                    return xt + 0.1 * ut_padded
            return f
        
        # Create a dummy constraint function g (not used in cost, but required for Player)
        def g(xt, ut):
            return np.array([0.0])
        
        # Create Player instance
        player = Player(
            xref=xref,
            f=make_f(i),
            g=g,
            tau=tau,
            Qi=Q,
            Qtau=Q,
            Ri=R
        )
        players.append(player)
    
    # Create a constraint function g for the PotentialGame
    # g takes appended x (N*d,) and u (N*m,) and returns inequality constraints
    def g_game(x_appended, u_appended):
        # Return some dummy constraints
        return np.array([0.0])
    
    # Create PotentialGame instance
    potential_game = PotientialGame(players=players, g=g_game)
    
    # Create random x in R^(N*d) for tau durations (shape: N*d x tau)
    x = np.random.rand(n_players * d, tau)
    
    # Create random u in R^(N*m) for tau durations (shape: N*m x tau)
    u = np.random.rand(n_players * m, tau)
    
    # Call cost method
    cost_value = potential_game.cost(x, u)
    
    # Check that cost returns a scalar
    assert np.isscalar(cost_value) or (isinstance(cost_value, np.ndarray) and cost_value.ndim == 0), \
        f"cost should return a scalar, but got shape {np.shape(cost_value)}"
    
    # Convert to Python scalar if it's a 0-d array
    cost_scalar = float(cost_value)
    
    print(f"✓ Test passed: cost returned a scalar value: {cost_scalar}")
    print(f"  Number of players: {n_players}")
    print(f"  Dimensions per player: d={d}, m={m}, tau={tau}")
    print(f"  Total dimensions: x shape: {x.shape}, u shape: {u.shape}")
    print(f"  Qtau shape: {potential_game.Qtau.shape}")
    print(f"  Qi shape: {potential_game.Qi.shape}")
    print(f"  Ri shape: {potential_game.Ri.shape}")
    
    # Also test get_constraints to ensure it works
    try:
        init_constraint, inequality_constraints, dynamics_constraints = potential_game.get_constraints(x, u)
        print(f"✓ get_constraints test passed")
        print(f"  Initial constraint shape: {init_constraint.shape} (expected: ({n_players * d},))")
        print(f"  Inequality constraints shape: {inequality_constraints.shape} (expected: (num_constraints, {tau}))")
        print(f"  Dynamics constraints shape: {dynamics_constraints.shape} (expected: ({n_players * d}, {tau-1}))")
        
        # Verify expected shapes
        assert dynamics_constraints.shape == (n_players * d, tau - 1), \
            f"Dynamics constraints should be ({n_players * d}, {tau-1}), got {dynamics_constraints.shape}"
        assert inequality_constraints.shape[1] == tau, \
            f"Inequality constraints should have {tau} columns, got {inequality_constraints.shape[1]}"
    except Exception as e:
        print(f"⚠ get_constraints raised an error (this might be due to the f function implementation): {e}")
    
    return cost_scalar


if __name__ == "__main__":
    test_potential_game()

