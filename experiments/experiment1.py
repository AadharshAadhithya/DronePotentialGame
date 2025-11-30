import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import sys
import os
import pandas as pd

# Add parent directory to path to allow importing modules from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solve import solve_game
from player import Player, PotientialGame

# ==========================================
# EXPERIMENT SETUP
# ==========================================

# Map Dimensions (for plotting)
MAP_X_LIMIT = 20.0
MAP_Y_LIMIT = 10.0

# Scenario: Head-on collision (Chicken Game)
# Player 1 moves Left->Right, Player 2 moves Right->Left
PLAYERS_SETUP = [
    {'start': (2.0, 5.0), 'goal': (18.0, 5.0), 'color': 'blue'},
    {'start': (18.0, 5.0), 'goal': (2.0, 5.0), 'color': 'red'}
]

OBSTACLES_SETUP = [] # Open road

RI_MULTIPLIERS = {
    "Low": 0.1,
    "Baseline": 1.0,
    "High": 10.0
}

# ==========================================
# GAME CREATION HELPER
# ==========================================

def create_custom_car_game(players_config, obstacles, ri_multiplier=1.0, tau=20, dt=0.2):
    """
    Creates a car game with specific Ri multiplier.
    Based on car.create_general_car_game but with adjustable Ri.
    """
    # Dimensions
    d = 5 # x = [p, q, theta, v, omega]
    m = 2 # u = [delta_v, delta_omega]
    
    n_players = len(players_config)
    
    # Cost matrices
    Qhat = np.diag(np.array([50.0, 10.0, 5.0, 5.0, 2.0]))
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    
    # BASELINE Ri
    Ri_base = np.diag(np.array([8.0, 4.0]))
    Ri = Ri_base * ri_multiplier
    
    # Dynamics function
    def car_dynamics(x, u):
        p, q, theta, v, omega = x
        delta_v, delta_omega = u
        
        p_next = p + dt * v * np.cos(theta)
        q_next = q + dt * v * np.sin(theta)
        theta_next = theta + dt * omega
        v_next = v + delta_v
        omega_next = omega + delta_omega
        
        return np.array([p_next, q_next, theta_next, v_next, omega_next])
    
    # Individual constraints
    def g_dummy(x, u):
        return np.array([0.0])
        
    players = []
    
    for cfg in players_config:
        start = cfg['start']
        goal = cfg['goal']
        
        # Create reference trajectory
        xref = np.zeros((d, tau))
        t_range = np.arange(tau, dtype=float)
        
        # Linear interpolation for position
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Angle
        target_theta = np.arctan2(dy, dx)
        
        # Reference Speed
        duration = (tau - 1) * dt
        ref_v = dist / duration if duration > 0 else 0.0
        
        # Steps 0 to tau-1
        s = t_range / (tau - 1)
        
        xref = xref.at[0, :].set(start[0] + s * dx)
        xref = xref.at[1, :].set(start[1] + s * dy)
        xref = xref.at[2, :].set(target_theta)
        xref = xref.at[3, :].set(ref_v) 
        xref = xref.at[4, :].set(0.0)
        
        p = Player(xref=xref, f=car_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
        players.append(p)
        
    # Joint constraints
    r_col = 1.5 # Collision radius (slightly larger than car)
    u_bound = np.array([0.2, 0.8]) # Adjusted bounds slightly? No keeping similar to car.py
    # car.py used [0.15, 0.75] in one place and [0.2, 0.8] in another?
    # car.py create_general_car_game used [0.15, 0.75]
    u_bound = np.array([0.15, 0.75])

    def g_constraints(x_joint, u_joint):
        constraints = []
        
        # 1. Collision avoidance
        for i in range(n_players):
            xi = x_joint[i*d:(i+1)*d]
            for j in range(i+1, n_players):
                xj = x_joint[j*d:(j+1)*d]
                dist = np.sqrt((xi[0] - xj[0])**2 + (xi[1] - xj[1])**2)
                constraints.append(-dist + r_col)
        
        # 2. Obstacle avoidance
        for i in range(n_players):
            xi = x_joint[i*d:(i+1)*d]
            for obs in obstacles:
                dist = np.sqrt((xi[0] - obs['x'])**2 + (xi[1] - obs['y'])**2)
                constraints.append(-dist + obs['r'])
                
        # 3. Positive velocity
        for i in range(n_players):
            xi = x_joint[i*d:(i+1)*d]
            constraints.append(-xi[3])
            
        # 4. Control bounds
        for i in range(n_players):
            ui = u_joint[i*m:(i+1)*m]
            constraints.extend(ui - u_bound)
            constraints.extend(-ui - u_bound)
            
        if not constraints:
            return np.array([0.0])
            
        return np.array(constraints)
        
    game = PotientialGame(players=players, g=g_constraints)
    game.type = 'car'
    game.obstacles_list = obstacles
    return game

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(x_sol, u_sol, game, dt=0.2):
    n = game.n
    d = 5
    m = 2
    tau = game.tau
    
    metrics = {}
    
    # 1. Min Distance between players
    # Assuming 2 players for this experiment
    x1 = x_sol[0:d, :]
    x2 = x_sol[d:2*d, :]
    dists = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
    metrics['min_distance'] = float(np.min(dists))
    
    # 2. Jerk (Smoothness) -> Mean change in control input
    # u is (N*m, tau). Since tau is usually controls at 0..tau-1 or similar.
    # Depending on implementation, u might be defined for t=0..tau-1.
    # Let's compute average difference between consecutive u's.
    # sum(|u_t - u_{t-1}|)
    total_jerk = 0.0
    
    # For each player
    for i in range(n):
        u_i = u_sol[i*m : (i+1)*m, :] # (2, tau)
        # diff along time axis
        u_diff = np.diff(u_i, axis=1) # (2, tau-1)
        # Norm of change vector at each step, summed
        jerk_i = np.sum(np.linalg.norm(u_diff, axis=0))
        total_jerk += jerk_i
        
    metrics['jerk'] = float(total_jerk / n) # Average per player
    
    # 3. Time to Goal
    # Check when they reach within goal_radius of their goal
    goal_radius = 1.0
    times_to_goal = []
    
    for i in range(n):
        # Goal from last point of reference trajectory or logic?
        # We know the goals from PLAYERS_SETUP.
        goal = PLAYERS_SETUP[i]['goal']
        
        # Check distance at each time step
        p_curr = x_sol[i*d : i*d+2, :]
        dist_to_goal = np.sqrt((p_curr[0] - goal[0])**2 + (p_curr[1] - goal[1])**2)
        
        # Find first index where dist < goal_radius
        reached_indices = np.where(dist_to_goal < goal_radius)[0]
        
        if len(reached_indices) > 0:
            # Time = index * dt
            t_goal = reached_indices[0] * dt
            times_to_goal.append(t_goal)
        else:
            # Did not reach
            times_to_goal.append(float('inf'))
            
    # Average time to goal (if inf, keep inf or max time?)
    # For the table, maybe list both or average valid ones?
    # "Do they slow down to let the other pass?" -> If one slows down, their time increases.
    # Let's store both.
    metrics['time_to_goal_p1'] = times_to_goal[0]
    metrics['time_to_goal_p2'] = times_to_goal[1]
    
    # Clean inf for CSV
    if metrics['time_to_goal_p1'] == float('inf'):
         metrics['time_to_goal_p1'] = tau * dt # Max time
    if metrics['time_to_goal_p2'] == float('inf'):
         metrics['time_to_goal_p2'] = tau * dt
         
    # 4. Evasion Start Time (Time of First Deviation)
    # Check when lateral deviation > threshold (e.g. 0.1m)
    # Players move along y=5.0 mostly.
    deviation_threshold = 0.1
    evasion_times = []
    
    for i in range(n):
        # Assuming straight line motion along x-axis at y=start_y is the reference "lazy" path
        # However, they are head-on. P1: (2,5)->(18,5). P2: (18,5)->(2,5).
        # So ideal path is y=5.0.
        start_y = PLAYERS_SETUP[i]['start'][1]
        
        # y trajectory
        y_traj = x_sol[i*d + 1, :] # q component
        
        # Deviation from start_y
        deviation = np.abs(y_traj - start_y)
        
        # First index where deviation > threshold
        dev_indices = np.where(deviation > deviation_threshold)[0]
        
        if len(dev_indices) > 0:
            evasion_times.append(dev_indices[0] * dt)
        else:
            evasion_times.append(float('nan')) # Never deviated enough?
            
    metrics['evasion_start_p1'] = evasion_times[0]
    metrics['evasion_start_p2'] = evasion_times[1]
    
    return metrics

# ==========================================
# PLOTTING
# ==========================================

def plot_map_setup(filename="experiment1_map.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAP_X_LIMIT)
    ax.set_ylim(0, MAP_Y_LIMIT)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Draw Start/Goals
    for p in PLAYERS_SETUP:
        # Start
        ax.plot(p['start'][0], p['start'][1], marker='o', color=p['color'], label=f"Start {p['color']}")
        ax.text(p['start'][0], p['start'][1]+0.5, "S", color=p['color'], ha='center')
        
        # Goal
        ax.plot(p['goal'][0], p['goal'][1], marker='x', markersize=10, color=p['color'], label=f"Goal {p['color']}")
        ax.text(p['goal'][0], p['goal'][1]+0.5, "G", color=p['color'], ha='center')
        
    ax.set_title("Experiment 1 Setup: Head-on Collision")
    plt.savefig(filename)
    plt.close()
    print(f"Map setup saved to {filename}")

def save_results(name, game, x_sol, u_sol, dt, output_dir):
    # 1. Static Trajectory Plot
    n = game.n
    d = 5
    tau = game.tau
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAP_X_LIMIT)
    ax.set_ylim(0, MAP_Y_LIMIT)
    ax.set_aspect('equal')
    ax.grid(True)
    
    for i in range(n):
        color = PLAYERS_SETUP[i]['color']
        traj = x_sol[i*d : i*d+2, :]
        ax.plot(traj[0, :], traj[1, :], '.-', color=color, label=f"Player {i+1}")
        # Mark every 5th point to see speed
        ax.scatter(traj[0, ::5], traj[1, ::5], color=color, s=10)
        
    ax.legend()
    ax.set_title(f"Trajectories - {name}")
    plt.savefig(os.path.join(output_dir, f"experiment1_traj_{name}.png"))
    plt.close()
    
    # 2. Animation (Optional, but requested)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAP_X_LIMIT)
    ax.set_ylim(0, MAP_Y_LIMIT)
    ax.set_aspect('equal')
    ax.grid(True)
    
    lines = [ax.plot([], [], 'o-', color=PLAYERS_SETUP[i]['color'])[0] for i in range(n)]
    
    def update(frame):
        for i in range(n):
            traj = x_sol[i*d : i*d+2, :frame+1]
            lines[i].set_data(traj[0], traj[1])
        return lines
        
    ani = FuncAnimation(fig, update, frames=tau, blit=True)
    ani.save(os.path.join(output_dir, f"experiment1_{name}.gif"), writer='pillow', fps=10)
    plt.close()

# ==========================================
# MAIN RUNNER
# ==========================================

def main():
    # Ensure artifacts directory exists
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiment1")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Plot Map
    plot_map_setup(os.path.join(artifacts_dir, "experiment1_map.png"))
    
    results = []
    
    for name, multiplier in RI_MULTIPLIERS.items():
        print(f"\nRunning Experiment: {name} (Ri x {multiplier})")
        
        # Create Game
        game = create_custom_car_game(PLAYERS_SETUP, OBSTACLES_SETUP, ri_multiplier=multiplier)
        
        # Solve
        # Using a fixed seed for reproducibility
        seed = 42
        x_sol, u_sol, res = solve_game(game, seed=seed)
        
        if not res.success and "acceptable" not in str(res.message).lower():
            print(f"Warning: Solver did not converge for {name}")
        
        # Metrics
        met = calculate_metrics(x_sol, u_sol, game)
        met['condition'] = name
        met['ri_multiplier'] = multiplier
        results.append(met)
        
        print(f"Metrics: {met}")
        
        # Save Plots
        save_results(name, game, x_sol, u_sol, dt=0.2, output_dir=artifacts_dir)
        
    # Save CSV
    df = pd.DataFrame(results)
    # Reorder columns
    cols = ['condition', 'ri_multiplier', 'min_distance', 'jerk', 'time_to_goal_p1', 'time_to_goal_p2', 'evasion_start_p1', 'evasion_start_p2']
    df = df[cols]
    df.to_csv(os.path.join(artifacts_dir, "experiment1.csv"), index=False)
    print(f"\nExperiment completed. Results saved to {artifacts_dir}/experiment1.csv")

if __name__ == "__main__":
    main()

