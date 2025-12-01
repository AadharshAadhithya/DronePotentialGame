import jax.numpy as np
from jax import config
import sys
import os

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

from cyipopt import minimize_ipopt
from jax import jit, grad, jacrev, jacfwd
import numpy as onp # Standard numpy for shapes and initial guess

# Add parent directory to path to import classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from car import create_car_game
from drone import create_drone_game

def solve_game(game, seed=None, noise_scale=0.01, warm_start=None):
    """
    Solves the potential game using IPOPT.
    """
    if seed is not None:
        onp.random.seed(seed)
        print(f"Solving with random seed: {seed}")
    # Extract dimensions
    n_players = game.n
    d = game.Qtau.shape[0] // n_players # state dimension per player (assuming equal)
    # Actually Qtau is (N*d, N*d). But d might be different per player? 
    # The code assumes all players have same d in block diagonal construction logic usually.
    # Let's use the full dimension from Qtau.
    full_d = game.Qtau.shape[0]
    full_m = game.Ri.shape[0]
    tau = game.tau
    
    n_x = full_d * tau
    n_u = full_m * tau
    
    print(f"Solving game with {n_players} players, tau={tau}")
    print(f"Total state dimension: {full_d}, Total control dimension: {full_m}")
    print(f"Optimization variables: {n_x + n_u}")
    
    # Helper to reshape z -> (x, u)
    def reshape_z(z):
        x_flat = z[:n_x]
        u_flat = z[n_x:]
        x = x_flat.reshape((full_d, tau))
        u = u_flat.reshape((full_m, tau))
        return x, u

    # Objective function
    def objective(z):
        x, u = reshape_z(z)
        return game.cost(x, u)

    # Equality constraints: Initial condition and Dynamics
    # Returns a flat array where each element must be 0
    def eq_constraints(z):
        x, u = reshape_z(z)
        init_c, _, dyn_c = game.get_constraints(x, u)
        # init_c: (full_d,)
        # dyn_c: (full_d, tau-1)
        
        # Flatten and concatenate
        return np.concatenate([init_c, dyn_c.flatten()])

    # Inequality constraints: g(x,u) <= 0
    # IPOPT expects g(x) >= 0 usually if type='ineq'.
    # cyipopt documentation says for 'ineq', constraint(x) >= 0.
    # Our game returns values that should be <= 0.
    # So we return -1 * constraint values.
    def ineq_constraints(z):
        x, u = reshape_z(z)
        _, ineq_c, _ = game.get_constraints(x, u)
        # ineq_c: (num_constraints, tau)
        
        # We want -ineq_c >= 0
        return -ineq_c.flatten()

    # JIT compile functions
    obj_jit = jit(objective)
    con_eq_jit = jit(eq_constraints)
    con_ineq_jit = jit(ineq_constraints)

    # Derivatives
    obj_grad = jit(grad(obj_jit))
    con_eq_jac = jit(jacfwd(con_eq_jit))
    con_ineq_jac = jit(jacfwd(con_ineq_jit))
    
    # Wrappers to convert JAX arrays to standard numpy arrays
    # This can prevent issues with low-level solvers like MUMPS
    def obj_wrapper(x):
        return onp.array(obj_jit(x))
    
    def obj_grad_wrapper(x):
        return onp.array(obj_grad(x))
        
    def con_eq_wrapper(x):
        return onp.array(con_eq_jit(x))
        
    def con_eq_jac_wrapper(x):
        return onp.array(con_eq_jac(x))
        
    def con_ineq_wrapper(x):
        return onp.array(con_ineq_jit(x))
        
    def con_ineq_jac_wrapper(x):
        return onp.array(con_ineq_jac(x))

    # Constraints structure for cyipopt
    cons = [
        {'type': 'eq', 'fun': con_eq_wrapper, 'jac': con_eq_jac_wrapper},
        {'type': 'ineq', 'fun': con_ineq_wrapper, 'jac': con_ineq_jac_wrapper}
    ]
    
    # Initial guess
    if warm_start is not None:
        print("Using provided warm start.")
        x0_ref, u0_ref = warm_start
        # Ensure they are correct type/shape if needed, but assuming correct coming from pf
    else:
        # Initialize with zeros or reference trajectory?
        # Using reference trajectory for x might be good.
        x0_ref = game.xref # (full_d, tau)
        u0_ref = np.zeros((full_m, tau))
        
    z0 = np.concatenate([x0_ref.flatten(), u0_ref.flatten()])
    
    # Add small noise to avoid singular points if any
    # Use standard numpy for random noise
    z0_np = onp.array(z0)
    if seed is not None:
        # Add noise only if seed is provided or always?
        # Prompt says "try different random intilizaionts".
        # So we use the seed to control this noise.
        z0_np = z0_np + noise_scale * onp.random.randn(z0_np.size)
    else:
        # Default behavior from before (though before we didn't check seed)
        z0_np = z0_np + noise_scale * onp.random.randn(z0_np.size)
    
    # Solve
    print("Starting optimization...")
    # options = {'disp': 5, 'max_iter': 500}
    options = {
        'disp': 5,
        'hessian_approximation': 'limited-memory',
        'tol': 1e-3,
        'acceptable_tol': 1e-2,
        'max_iter': 1000
    }
    
    # Note: we are not providing Hessian, so IPOPT will approximate it (L-BFGS usually or internal approximation)
    # We need to verify if 'hess' is required. If not provided, it might warn or use approx.
    
    res = minimize_ipopt(obj_wrapper, jac=obj_grad_wrapper, x0=z0_np, constraints=cons, options=options)
    
    print("Optimization finished.")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Objective value: {res.fun}")
    
    # Extract solution
    z_sol = res.x
    x_sol, u_sol = reshape_z(z_sol)
    
    return x_sol, u_sol, res

if __name__ == "__main__":
    # Choose game type: 'car' or 'drone'
    GAME_TYPE = 'drone'
    
    print(f"Creating {GAME_TYPE} game...")
    if GAME_TYPE == 'car':
        game = create_car_game()
    elif GAME_TYPE == 'drone':
        game = create_drone_game()
    else:
        raise ValueError(f"Unknown game type: {GAME_TYPE}")
    
    # Try multiple random seeds (just one for now)
    seeds = [0]
    
    import matplotlib.pyplot as plt
    import math
    from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
    
    num_plots = len(seeds)
    cols = 1
    rows = 1
    
    fig = plt.figure(figsize=(10, 8))
    
    # Extract reference for plotting (same for all)
    xref = game.xref
    d = game.Qtau.shape[0] // game.n
    xref1 = xref[0:d, :]
    xref2 = xref[d:2*d, :]

    for i, seed in enumerate(seeds):
        # Setup subplot
        if GAME_TYPE == 'drone':
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        else:
            ax = fig.add_subplot(rows, cols, i+1)
            
        print(f"\n{'='*50}")
        print(f"Run {i+1}/{len(seeds)} with seed {seed}")
        print(f"{'='*50}")
        
        x_sol, u_sol, res = solve_game(game, seed=seed, noise_scale=0.1)
        
        # Check if solved successfully or to acceptable level
        is_acceptable = False
        if hasattr(res, 'message'):
            msg = str(res.message)
            if "acceptable" in msg.lower():
                is_acceptable = True

        if res.success or is_acceptable:
            print(f"Solution found! (Cost: {res.fun:.4f})")
            
            n_players = game.n
            d = game.Qtau.shape[0] // n_players
            x1 = x_sol[0:d, :]
            x2 = x_sol[d:2*d, :]
            
            # Calculate min distance
            if GAME_TYPE == 'drone':
                dists = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 + (x1[2] - x2[2])**2)
            else:
                dists = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
            min_dist = np.min(dists)
            print(f"Min dist: {min_dist:.4f}")
            
            # Plot trajectories
            if GAME_TYPE == 'drone':
                ax.plot(x1[0, :], x1[1, :], x1[2, :], 'b.-', label='P1')
                ax.plot(x2[0, :], x2[1, :], x2[2, :], 'r.-', label='P2')
                
                # Start/End
                ax.scatter(x1[0, 0], x1[1, 0], x1[2, 0], c='b', marker='o')
                ax.scatter(x1[0, -1], x1[1, -1], x1[2, -1], c='b', marker='x')
                ax.scatter(x2[0, 0], x2[1, 0], x2[2, 0], c='r', marker='o')
                ax.scatter(x2[0, -1], x2[1, -1], x2[2, -1], c='r', marker='x')
                
                # Plot Obstacle (Cylinder)
                if hasattr(game, 'obstacle'):
                    obs = game.obstacle
                    # Draw cylinder wireframe
                    z_line = np.linspace(0, 20, 10) # Height
                    theta_line = np.linspace(0, 2*np.pi, 20)
                    theta_grid, z_grid = np.meshgrid(theta_line, z_line)
                    x_grid = obs['pos'][0] + obs['radius'] * np.cos(theta_grid)
                    y_grid = obs['pos'][1] + obs['radius'] * np.sin(theta_grid)
                    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='k')
                    
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
            else:
                ax.plot(x1[0, :], x1[1, :], 'b.-', label='P1')
                ax.plot(x2[0, :], x2[1, :], 'r.-', label='P2')
                
                # Mark start/end
                ax.plot(x1[0, 0], x1[1, 0], 'bo', markersize=7)
                ax.plot(x1[0, -1], x1[1, -1], 'bx', markersize=7)
                ax.plot(x2[0, 0], x2[1, 0], 'ro', markersize=7)
                ax.plot(x2[0, -1], x2[1, -1], 'rx', markersize=7)
                
                # Visualize collision radius
                min_idx = np.argmin(dists)
                circle1 = plt.Circle((x1[0, min_idx], x1[1, min_idx]), 1.5, color='b', fill=False, linestyle=':', alpha=0.3)
                circle2 = plt.Circle((x2[0, min_idx], x2[1, min_idx]), 1.5, color='r', fill=False, linestyle=':', alpha=0.3)
                ax.add_patch(circle1)
                ax.add_patch(circle2)
                ax.text(x1[0, min_idx], x1[1, min_idx], f"d={min_dist:.2f}", fontsize=8)
                
                # Plot Obstacle if it exists
                if hasattr(game, 'obstacle'):
                    obs = game.obstacle
                    obs_circle = plt.Circle((obs['pos'][0], obs['pos'][1]), obs['radius'], 
                                          color='k', alpha=0.2, label='Obstacle')
                    ax.add_patch(obs_circle)
        else:
            print(f"Failed to solve with seed {seed}")
            ax.text(0.5, 0.5, 0.5, "Failed", ha='center', va='center') if GAME_TYPE == 'drone' else ax.text(0.5, 0.5, "Failed", ha='center', va='center', transform=ax.transAxes)
            
        # Plot Reference
        if GAME_TYPE == 'drone':
            ax.plot(xref1[0, :], xref1[1, :], xref1[2, :], 'b--', alpha=0.3, label='Ref P1')
            ax.plot(xref2[0, :], xref2[1, :], xref2[2, :], 'r--', alpha=0.3, label='Ref P2')
        else:
            ax.plot(xref1[0, :], xref1[1, :], 'b--', alpha=0.3, label='Ref P1')
            ax.plot(xref2[0, :], xref2[1, :], 'r--', alpha=0.3, label='Ref P2')
        
        ax.set_title(f'Seed {seed}')
        if GAME_TYPE != 'drone':
            ax.grid(True)
            ax.axis('equal')
        
        if i == 0:
            ax.legend()
    
    fig.suptitle(f'{GAME_TYPE.capitalize()} Game Trajectories', fontsize=16)
    
    output_file = f'trajectories_{GAME_TYPE}.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

