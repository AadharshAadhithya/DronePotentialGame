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
# Adjusted for the closer start positions (-0.5 to 0.5)
MAP_X_LIMIT = 1.0
MAP_Y_LIMIT = 1.0
MAP_Z_LIMIT = 3.0
FIXED_HEIGHT = 2.0

# Scenario: Head-on collision (Chicken Game)
# Player 1 moves Left->Right (-x to +x), Player 2 moves Right->Left (+x to -x)
# Start and Goal positions updated as requested: -0.5 and 0.5 on x-axis.
PLAYERS_SETUP = [
    {'start': (-0.5, 0.0, FIXED_HEIGHT), 'goal': (0.5, 0.0, FIXED_HEIGHT), 'color': 'blue'},
    {'start': (0.5, 0.0, FIXED_HEIGHT), 'goal': (-0.5, 0.0, FIXED_HEIGHT), 'color': 'red'}
]

OBSTACLES_SETUP = [] # Open road

RI_MULTIPLIERS = {
    "Low": 0.1,
    "Baseline": 1.0,
    "High": 10.0,
    "Asymmetric": [10.0, 0.1] # P1 Polite, P2 Aggressive
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
                             10.0, 10.0,        # angles
                             5.0,               # v
                             5.0, 2.0])         # rates
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
        
        # Angles (simplified: pointing to goal in yaw, flat pitch)
        target_theta = np.arctan2(dy, dx)
        target_phi = np.arctan2(dz, np.sqrt(dx**2 + dy**2)) 
        
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
    
    # 1. Min Distance between players
    x1 = x_sol[0:d, :]
    x2 = x_sol[d:2*d, :]
    # 3D distance
    dists = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 + (x1[2] - x2[2])**2)
    metrics['min_distance'] = float(np.min(dists))
    
    # 2. Jerk (Smoothness) -> Mean change in control input
    total_jerk = 0.0
    
    for i in range(n):
        u_i = u_sol[i*m : (i+1)*m, :] # (3, tau)
        u_diff = np.diff(u_i, axis=1) # (3, tau-1)
        jerk_i = np.sum(np.linalg.norm(u_diff, axis=0))
        total_jerk += jerk_i
        
    metrics['jerk'] = float(total_jerk / n)
    
    # 3. Time to Goal
    goal_radius = 0.2 # Reduced radius for small scale
    times_to_goal = []
    
    for i in range(n):
        goal = PLAYERS_SETUP[i]['goal']
        
        p_curr = x_sol[i*d : i*d+3, :] # x, y, z
        dist_to_goal = np.sqrt((p_curr[0] - goal[0])**2 + (p_curr[1] - goal[1])**2 + (p_curr[2] - goal[2])**2)
        
        reached_indices = np.where(dist_to_goal < goal_radius)[0]
        
        if len(reached_indices) > 0:
            t_goal = reached_indices[0] * dt
            times_to_goal.append(t_goal)
        else:
            times_to_goal.append(float('inf'))
            
    metrics['time_to_goal_p1'] = times_to_goal[0]
    metrics['time_to_goal_p2'] = times_to_goal[1]
    
    if metrics['time_to_goal_p1'] == float('inf'):
         metrics['time_to_goal_p1'] = tau * dt
    if metrics['time_to_goal_p2'] == float('inf'):
         metrics['time_to_goal_p2'] = tau * dt
         
    # 4. Evasion Start Time (Deviation from straight line)
    # Straight line is y=0
    deviation_threshold = 0.05 # Reduced threshold for small scale
    evasion_times = []
    
    for i in range(n):
        start_y = PLAYERS_SETUP[i]['start'][1]
        y_traj = x_sol[i*d + 1, :] # q component (y)
        
        deviation = np.abs(y_traj - start_y)
        dev_indices = np.where(deviation > deviation_threshold)[0]
        
        if len(dev_indices) > 0:
            evasion_times.append(dev_indices[0] * dt)
        else:
            evasion_times.append(float('nan'))
            
    metrics['evasion_start_p1'] = evasion_times[0]
    metrics['evasion_start_p2'] = evasion_times[1]
    
    # 5. Max Lateral Deviation
    devs = []
    for i in range(n):
        start_y = PLAYERS_SETUP[i]['start'][1]
        y_traj = x_sol[i*d + 1, :]
        max_dev = np.max(np.abs(y_traj - start_y))
        devs.append(max_dev)
        
    metrics['max_dev_p1'] = float(devs[0])
    metrics['max_dev_p2'] = float(devs[1])
    
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
        ax.scatter(start[0], start[1], start[2], color=color, marker='o', s=100, label=f"Start {color}")
        ax.text(start[0], start[1], start[2], "S", color=color)
        
        # Goal
        ax.scatter(goal[0], goal[1], goal[2], color=color, marker='x', s=100, label=f"Goal {color}")
        ax.text(goal[0], goal[1], goal[2], "G", color=color)
        
        # Ideal path
        ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], '--', color=color, alpha=0.3)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Experiment 1 (Drone) Setup: Head-on Collision")
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
    ax.set_title(f"Trajectories - {name} (Drone)")
    plt.savefig(os.path.join(output_dir, f"experiment1_drone_traj_{name}.png"))
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
    ani.save(os.path.join(output_dir, f"experiment1_drone_{name}.gif"), writer='pillow', fps=10)
    plt.close()

def save_detailed_csvs(name, x_sol, u_sol, dt, output_dir):
    # Format: drone_id,t,x,y,z,yaw
    n = 2 
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
    
    # Use specific filenames requested, appending condition for non-Baseline to avoid overwrite
    if name == "Baseline":
        df_x.to_csv(os.path.join(output_dir, "drone_exp1_x.csv"), index=False, float_format='%.4f')
    else:
        df_x.to_csv(os.path.join(output_dir, f"drone_exp1_x_{name}.csv"), index=False, float_format='%.4f')

    # Prepare U CSV
    # Format: drone_id,t,u1,u2,u3
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
    
    # Assuming "drone_exp2_u.csv" in prompt was typo for "drone_exp1_u.csv"
    if name == "Baseline":
        df_u.to_csv(os.path.join(output_dir, "drone_exp1_u.csv"), index=False, float_format='%.4f')
    else:
        df_u.to_csv(os.path.join(output_dir, f"drone_exp1_u_{name}.csv"), index=False, float_format='%.4f')

# ==========================================
# MAIN RUNNER
# ==========================================

def main():
    # Ensure artifacts directory exists
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiment1_drone")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Plot Map
    plot_map_setup(os.path.join(artifacts_dir, "experiment1_drone_map.png"))
    
    results = []
    
    for name, multiplier in RI_MULTIPLIERS.items():
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
        results.append(met)
        
        print(f"Metrics: {met}")
        
        # Save Plots
        save_results(name, game, x_sol, u_sol, dt=dt_val, output_dir=artifacts_dir)
        
        # Save Detailed CSVs
        save_detailed_csvs(name, x_sol, u_sol, dt=dt_val, output_dir=artifacts_dir)
        
    # Save CSV
    df = pd.DataFrame(results)
    cols = ['condition', 'ri_multiplier', 'min_distance', 'jerk', 'time_to_goal_p1', 'time_to_goal_p2', 'evasion_start_p1', 'evasion_start_p2', 'max_dev_p1', 'max_dev_p2']
    df = df[cols]
    df.to_csv(os.path.join(artifacts_dir, "experiment1_drone.csv"), index=False)
    print(f"\nExperiment completed. Results saved to {artifacts_dir}/experiment1_drone.csv")

if __name__ == "__main__":
    main()
