import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
MAP_X_LIMIT = 1.0
MAP_Y_LIMIT = 1.0
MAP_Z_LIMIT = 2.0
FIXED_HEIGHT = 0.5

# Scenario: 4 Drones Swapping Places
# Drones start at (0.5,0), (-0.5,0), (0,0.5), (0,-0.5) and go to opposite side
# Adjusted slightly to match user request "x=0.5, x=-0.5, y=0.5, y=-0.5"
# Assuming they are on axes.
# 1. (+0.5, 0) -> (-0.5, 0)
# 2. (-0.5, 0) -> (+0.5, 0)
# 3. (0, +0.5) -> (0, -0.5)
# 4. (0, -0.5) -> (0, +0.5)

PLAYERS_SETUP = [
    {'start': (0.5, 0.0, FIXED_HEIGHT), 'goal': (-0.5, 0.0, FIXED_HEIGHT), 'color': 'blue'},
    {'start': (-0.5, 0.0, FIXED_HEIGHT), 'goal': (0.5, 0.0, FIXED_HEIGHT), 'color': 'red'},
    {'start': (0.0, 0.5, FIXED_HEIGHT), 'goal': (0.0, -0.5, FIXED_HEIGHT), 'color': 'green'},
    {'start': (0.0, -0.5, FIXED_HEIGHT), 'goal': (0.0, 0.5, FIXED_HEIGHT), 'color': 'purple'}
]

OBSTACLES_SETUP = [] # Open road

RI_MULTIPLIERS = {
    "Low": 0.1,
    "Baseline": 1.0,
    "High": 10.0,
    "Asymmetric": [10.0, 0.1, 1.0, 1.0] # Mixed behavior
}

# ==========================================
# GAME CREATION HELPER
# ==========================================

def create_custom_drone_game(players_config, obstacles, ri_multiplier=1.0, tau=40, dt=0.1):
    """
    Creates a drone game with specific Ri multiplier.
    """
    # Dimensions
    d = 8   # state dim per drone [p, q, r, theta, phi, v, omega_theta, omega_phi]
    m = 3   # control dim per drone [delta_v, delta_omega_theta, delta_omega_phi]
    
    n_players = len(players_config)
    
    # Cost matrices
    diag_entries = np.array([50.0, 50.0, 50.0,  # pos
                             10.0, 50.0,        # angles (high phi cost)
                             5.0,               # v
                             20.0, 2.0])        # rates: high yaw rate cost
    Qhat = np.diag(diag_entries)
    Qi   = 0.6 * Qhat
    Qtau = 100 * Qhat
    
    # BASELINE Ri
    Ri_base = np.diag(np.array([8.0, 4.0, 4.0]))
    
    # Dynamics function
    def drone_dynamics_local(x, u):
        p, q, r, theta, phi, v, omega_theta, omega_phi = x
        delta_v, delta_omega_theta, delta_omega_phi = u
        
        p_next = p + dt * v * np.cos(theta) * np.cos(phi)
        q_next = q + dt * v * np.sin(theta) * np.cos(phi)
        r_next = r + dt * v * np.sin(phi)
        theta_next = theta + dt * omega_theta
        phi_next   = phi   + dt * omega_phi
        v_next           = v           + delta_v
        omega_theta_next = omega_theta + delta_omega_theta
        omega_phi_next   = omega_phi   + delta_omega_phi
        
        return np.array([p_next, q_next, r_next,
                         theta_next, phi_next,
                         v_next, omega_theta_next, omega_phi_next])
    
    # Individual constraints
    def g_dummy(x, u):
        return np.array([0.0])
        
    players = []
    
    for i, cfg in enumerate(players_config):
        start = cfg['start']
        goal = cfg['goal']
        
        # Calculate Ri for this player
        if isinstance(ri_multiplier, list):
            # Asymmetric case
            mult = ri_multiplier[i % len(ri_multiplier)]
        else:
            # Symmetric case
            mult = ri_multiplier
            
        Ri = Ri_base * mult
        
        # Create reference trajectory
        xref = np.zeros((d, tau))
        t_range = np.arange(tau, dtype=float)
        
        # Linear interpolation for position
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dz = goal[2] - start[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Angles: Align yaw (theta) with direction of travel
        target_theta = np.arctan2(dy, dx)
        target_phi = 0.0   # Force 0 pitch ref
        
        # Reference Speed
        duration = (tau - 1) * dt
        ref_v = dist / duration if duration > 0 else 0.0
        
        # Steps 0 to tau-1
        s = t_range / (tau - 1)
        
        xref = xref.at[0, :].set(start[0] + s * dx)
        xref = xref.at[1, :].set(start[1] + s * dy)
        xref = xref.at[2, :].set(start[2] + s * dz)
        xref = xref.at[3, :].set(target_theta)
        xref = xref.at[4, :].set(target_phi)
        xref = xref.at[5, :].set(ref_v)
        xref = xref.at[6, :].set(0.0) # omega_theta
        xref = xref.at[7, :].set(0.0) # omega_phi
        
        p = Player(xref=xref, f=drone_dynamics_local, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
        players.append(p)
        
    # Joint constraints
    r_col = 0.2
    u_bound = np.array([0.15, 0.75, 0.75])

    def g_constraints(x_joint, u_joint):
        constraints = []
        
        # 1. Collision avoidance
        for i in range(n_players):
            xi = x_joint[i*d:(i+1)*d]
            for j in range(i+1, n_players):
                xj = x_joint[j*d:(j+1)*d]
                # 3D distance
                dist = np.sqrt((xi[0] - xj[0])**2 + (xi[1] - xj[1])**2 + (xi[2] - xj[2])**2)
                constraints.append(-dist + r_col)
        
        # 2. Obstacle avoidance (if any)
        # ... skipped as OBSTACLES_SETUP is empty
                
        # 3. Positive velocity
        for i in range(n_players):
            xi = x_joint[i*d:(i+1)*d]
            constraints.append(-xi[5]) # v is index 5
            
        # 4. Control bounds
        for i in range(n_players):
            ui = u_joint[i*m:(i+1)*m]
            constraints.extend(ui - u_bound)
            constraints.extend(-ui - u_bound)
            
        if not constraints:
            return np.array([0.0])
            
        return np.array(constraints)
        
    game = PotientialGame(players=players, g=g_constraints)
    game.type = 'drone'
    game.configs = players_config # Save for plotting
    game.obstacles_list = obstacles
    return game

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(x_sol, u_sol, game, dt=0.1):
    n = game.n
    d = 8
    m = 3
    tau = game.tau
    
    metrics = {}
    
    # 1. Min Distance between ANY pair of players
    min_dist_global = float('inf')
    for i in range(n):
        xi = x_sol[i*d:(i+1)*d, :]
        for j in range(i+1, n):
            xj = x_sol[j*d:(j+1)*d, :]
            dists = np.sqrt((xi[0]-xj[0])**2 + (xi[1]-xj[1])**2 + (xi[2]-xj[2])**2)
            min_dist_pair = float(np.min(dists))
            if min_dist_pair < min_dist_global:
                min_dist_global = min_dist_pair
                
    metrics['min_distance'] = min_dist_global
    
    # 2. Jerk (Smoothness) -> Mean change in control input
    total_jerk = 0.0
    
    for i in range(n):
        u_i = u_sol[i*m : (i+1)*m, :] # (3, tau)
        u_diff = np.diff(u_i, axis=1) # (3, tau-1)
        jerk_i = np.sum(np.linalg.norm(u_diff, axis=0))
        total_jerk += jerk_i
        
    metrics['jerk'] = float(total_jerk / n)
    
    # 3. Time to Goal (Avg)
    goal_radius = 0.2
    total_time = 0.0
    
    for i in range(n):
        goal = PLAYERS_SETUP[i]['goal']
        p_curr = x_sol[i*d : i*d+3, :]
        dist_to_goal = np.sqrt((p_curr[0] - goal[0])**2 + (p_curr[1] - goal[1])**2 + (p_curr[2] - goal[2])**2)
        reached_indices = np.where(dist_to_goal < goal_radius)[0]
        
        if len(reached_indices) > 0:
            total_time += reached_indices[0] * dt
        else:
            total_time += tau * dt
            
    metrics['avg_time_to_goal'] = float(total_time / n)
         
    # 4. Max Deviation from Straight Line (3D)
    # Each player moves from start to goal. 
    # We project position onto vector perpendicular to start->goal vector.
    max_dev_global = 0.0
    
    for i in range(n):
        start = np.array(PLAYERS_SETUP[i]['start'])
        goal = np.array(PLAYERS_SETUP[i]['goal'])
        
        # Vector along path
        path_vec = goal - start
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6: continue
        path_dir = path_vec / path_len
        
        # Trajectory
        p_traj = x_sol[i*d : i*d+3, :] # (3, tau)
        
        # Vector from start to current pos
        rel_pos = p_traj - start.reshape(3,1) # (3, tau)
        
        # Projection onto path direction
        proj_len = np.dot(path_dir, rel_pos) # (tau,)
        proj_vec = np.outer(path_dir, proj_len) # (3, tau)
        
        # Perpendicular component
        perp_vec = rel_pos - proj_vec
        perp_dist = np.linalg.norm(perp_vec, axis=0) # (tau,)
        
        max_dev_i = np.max(perp_dist)
        if max_dev_i > max_dev_global:
            max_dev_global = max_dev_i
            
    metrics['max_deviation'] = float(max_dev_global)
    
    return metrics

# ==========================================
# PLOTTING AND SAVING
# ==========================================

def plot_map_setup(filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-MAP_X_LIMIT, MAP_X_LIMIT)
    ax.set_ylim(-MAP_Y_LIMIT, MAP_Y_LIMIT)
    ax.set_zlim(0, MAP_Z_LIMIT)
    
    for i, p in enumerate(PLAYERS_SETUP):
        start = p['start']
        goal = p['goal']
        color = p['color']
        
        # Start
        ax.scatter(start[0], start[1], start[2], color=color, marker='o', s=100, label=f"Start {i+1}")
        ax.text(start[0], start[1], start[2], f"S{i+1}", color=color)
        
        # Goal
        ax.scatter(goal[0], goal[1], goal[2], color=color, marker='x', s=100, label=f"Goal {i+1}")
        ax.text(goal[0], goal[1], goal[2], f"G{i+1}", color=color)
        
        # Ideal path
        ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], '--', color=color, alpha=0.3)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Experiment 1 (4 Drones) Setup")
    ax.legend()
    
    plt.savefig(filename)
    plt.close()
    print(f"Map setup saved to {filename}")

def save_results(name, game, x_sol, u_sol, dt, output_dir):
    n = game.n
    d = 8
    tau = game.tau
    
    # 1. Static Trajectory Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-MAP_X_LIMIT, MAP_X_LIMIT)
    ax.set_ylim(-MAP_Y_LIMIT, MAP_Y_LIMIT)
    ax.set_zlim(0, MAP_Z_LIMIT)
    
    for i in range(n):
        color = PLAYERS_SETUP[i]['color']
        traj = x_sol[i*d : i*d+3, :] # x, y, z
        
        ax.plot(traj[0, :], traj[1, :], traj[2, :], '.-', color=color, label=f"Player {i+1}")
        # Start/End
        ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0], color=color, marker='o')
        ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1], color=color, marker='x')
        
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Trajectories - {name} (4 Drones)")
    plt.savefig(os.path.join(output_dir, f"experiment1_4drone_traj_{name}.png"))
    plt.close()
    
    # 2. Animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-MAP_X_LIMIT, MAP_X_LIMIT)
    ax.set_ylim(-MAP_Y_LIMIT, MAP_Y_LIMIT)
    ax.set_zlim(0, MAP_Z_LIMIT)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Animation - {name}")
    
    lines = [ax.plot([], [], [], 'o-', color=PLAYERS_SETUP[i]['color'])[0] for i in range(n)]
    
    def update(frame):
        for i in range(n):
            traj = x_sol[i*d : i*d+3, :frame+1]
            lines[i].set_data(traj[0], traj[1])
            lines[i].set_3d_properties(traj[2])
        return lines
        
    ani = FuncAnimation(fig, update, frames=tau, blit=True)
    ani.save(os.path.join(output_dir, f"experiment1_4drone_{name}.gif"), writer='pillow', fps=10)
    plt.close()

def save_detailed_csvs(name, x_sol, u_sol, dt, output_dir):
    # Format: drone_id,t,x,y,z,yaw
    n = 4
    d = 8
    m = 3
    tau = x_sol.shape[1]
    
    # Prepare X CSV
    x_data = []
    for i in range(n):
        drone_id = f"cf{i+1}"
        for t in range(tau):
            time = t * dt
            # State
            idx = i * d
            px = x_sol[idx + 0, t]
            py = x_sol[idx + 1, t]
            pz = x_sol[idx + 2, t]
            yaw = x_sol[idx + 3, t]
            
            x_data.append({
                'drone_id': drone_id,
                't': time,
                'x': px,
                'y': py,
                'z': pz,
                'yaw': yaw
            })
            
    df_x = pd.DataFrame(x_data)
    
    if name == "Baseline":
        df_x.to_csv(os.path.join(output_dir, "drone_exp1_4d_x.csv"), index=False, float_format='%.4f')
    else:
        df_x.to_csv(os.path.join(output_dir, f"drone_exp1_4d_x_{name}.csv"), index=False, float_format='%.4f')

    # Prepare U CSV
    u_data = []
    for i in range(n):
        drone_id = f"cf{i+1}"
        for t in range(tau):
            time = t * dt
            # Control
            idx = i * m
            u1 = u_sol[idx + 0, t]
            u2 = u_sol[idx + 1, t]
            u3 = u_sol[idx + 2, t]
            
            u_data.append({
                'drone_id': drone_id,
                't': time,
                'u1': u1,
                'u2': u2,
                'u3': u3
            })
            
    df_u = pd.DataFrame(u_data)
    
    if name == "Baseline":
        df_u.to_csv(os.path.join(output_dir, "drone_exp1_4d_u.csv"), index=False, float_format='%.4f')
    else:
        df_u.to_csv(os.path.join(output_dir, f"drone_exp1_4d_u_{name}.csv"), index=False, float_format='%.4f')

# ==========================================
# MAIN RUNNER
# ==========================================

def run_single_scenario(name, multiplier, artifacts_dir):
    print(f"\nRunning Experiment: {name} (Ri x {multiplier})")
    
    # Create Game
    dt_val = 0.1
    game = create_custom_drone_game(PLAYERS_SETUP, OBSTACLES_SETUP, ri_multiplier=multiplier, tau=40, dt=dt_val)
    
    # Solve
    seed = 42
    x_sol, u_sol, res = solve_game(game, seed=seed)
    
    if not res.success and "acceptable" not in str(res.message).lower():
        print(f"Warning: Solver did not converge for {name}")
    
    # Metrics
    met = calculate_metrics(x_sol, u_sol, game, dt=dt_val)
    met['condition'] = name
    met['ri_multiplier'] = multiplier
    
    print(f"Metrics: {met}")
    
    # Save Plots
    save_results(name, game, x_sol, u_sol, dt=dt_val, output_dir=artifacts_dir)
    
    # Save Detailed CSVs
    save_detailed_csvs(name, x_sol, u_sol, dt=dt_val, output_dir=artifacts_dir)
    
    # Save partial metrics for parallel execution
    res_df = pd.DataFrame([met])
    res_file = os.path.join(artifacts_dir, f"metrics_{name}.csv")
    res_df.to_csv(res_file, index=False)
    
    return met

def main():
    # Ensure artifacts directory exists
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiment1_4drone")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Plot Map
    plot_map_setup(os.path.join(artifacts_dir, "experiment1_4drone_map.png"))
    
    # Check for command line argument
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        if scenario_name in RI_MULTIPLIERS:
            run_single_scenario(scenario_name, RI_MULTIPLIERS[scenario_name], artifacts_dir)
            return
        else:
             print(f"Scenario {scenario_name} not found. Running all.")

    results = []
    
    for name, multiplier in RI_MULTIPLIERS.items():
        met = run_single_scenario(name, multiplier, artifacts_dir)
        results.append(met)
        
    # Save CSV
    df = pd.DataFrame(results)
    cols = ['condition', 'ri_multiplier', 'min_distance', 'jerk', 'avg_time_to_goal', 'max_deviation']
    df = df[cols]
    df.to_csv(os.path.join(artifacts_dir, "experiment1_4drone.csv"), index=False)
    print(f"\nExperiment completed. Results saved to {artifacts_dir}/experiment1_4drone.csv")

if __name__ == "__main__":
    main()

