
import jax.numpy as np
import numpy as onp
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solve import solve_game
from experiments.experiment1_4drone import create_custom_drone_game, PLAYERS_SETUP, OBSTACLES_SETUP

def debug_quick_4d():
    print("Running QUICK Debug Experiment (4 Drones)")
    
    # SPEED OPTIMIZATION: Reduced horizon
    tau = 15 
    dt_val = 0.1
    
    # Use ALL 4 players
    players_config = PLAYERS_SETUP
    
    print(f"Setup: {len(players_config)} players, tau={tau}")
    
    game = create_custom_drone_game(players_config, OBSTACLES_SETUP, ri_multiplier=0.1, tau=tau, dt=dt_val)
    
    # Solve
    print("\nSolving...")
    x_sol, u_sol, res = solve_game(game, seed=42)
    
    print(f"Solver Success: {res.success}")
    
    # --- PLOTTING ---
    print("\nGenerating Debug Plot...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    d = 8
    n = 4
    
    for i in range(n):
        x_i = x_sol[i*d:(i+1)*d, :]
        color = PLAYERS_SETUP[i]['color']
        
        # Plot Trajectory
        ax.plot(x_i[0, :], x_i[1, :], x_i[2, :], '.-', color=color, label=f'P{i+1}')
        
        # Start/End
        ax.scatter(x_i[0, 0], x_i[1, 0], x_i[2, 0], color=color, s=50, marker='o')
        ax.scatter(x_i[0, -1], x_i[1, -1], x_i[2, -1], color=color, s=50, marker='x')
        
        # Yaw Arrows (Quiver)
        step = 3
        for t in range(0, tau, step):
            px, py, pz = x_i[0, t], x_i[1, t], x_i[2, t]
            yaw = x_i[3, t]
            
            length = 0.1
            dx = length * np.cos(yaw)
            dy = length * np.sin(yaw)
            
            ax.quiver(px, py, pz, dx, dy, 0, color=color, alpha=0.5, length=0.1, arrow_length_ratio=0.5)
            
        # Displacement check
        disp = np.linalg.norm(x_i[0:3, -1] - x_i[0:3, 0])
        print(f"Player {i+1} ({color}) Displacement: {disp:.4f}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Debug: 4 Drones (tau={tau})\nArrows show Yaw")
    ax.legend()
    
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(0, 1.0)
    
    output_file = "debug_drone_4d_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    debug_quick_4d()
