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
from pf import get_coarse_estimates, dbscan_from_dist_matrix

# ==========================================
# EXPERIMENT 2 SETUP: The Bottleneck Capacity
# ==========================================

# Map Dimensions
MAP_X_LIMIT = 20.0
MAP_Y_LIMIT = 10.0
CORRIDOR_Y_MIN = 4.0
CORRIDOR_Y_MAX = 6.0

# Define the bottleneck obstacles
def create_bottleneck_obstacles():
    obstacles = []
    # Two massive obstacles to create a gap between y=3.5 and y=6.5 at x=10
    # Top Pillar (Center (10, 9.0), Radius 2.5) -> Bottom edge at y=6.5
    obstacles.append({'x': 10.0, 'y': 9.0, 'r': 2.5})
    
    # Bottom Pillar (Center (10, 1.0), Radius 2.5) -> Top edge at y=3.5
    obstacles.append({'x': 10.0, 'y': 1.0, 'r': 2.5})
    
    return obstacles

OBSTACLES_SETUP = create_bottleneck_obstacles()

# Variable: Number of Agents (Density)
# All agents start on left (x=2) and want to go right (x=18)
DENSITIES = {
    "2_Agents": 2,
    "3_Agents": 3,
    "4_Agents": 4,
    "Ambulance": 4,
    "Blocked_Ambulance": 4 
}

def get_players_config_funnel(n_agents, mode="Uniform"):
    players = []
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    if mode == "Blocked_Ambulance":
        # 4 Agent Scenario: Boxed In
        # P1: Slow Car (Blue) - Center Lane Ahead
        players.append({
            'start': (4.0, 5.0),
            'goal': (18.0, 5.0),
            'color': 'blue',
            'is_ambulance': False,
            'is_slow': True
        })
        
        # P2: Ambulance (Magenta) - Center Lane Behind
        players.append({
            'start': (0.0, 5.0),
            'goal': (18.0, 5.0),
            'color': 'magenta',
            'is_ambulance': True,
            'is_slow': False
        })
        
        # P3: Traffic (Green) - Top Lane
        players.append({
            'start': (2.0, 7.0),
            'goal': (18.0, 7.0),
            'color': 'green',
            'is_ambulance': False,
            'is_slow': False
        })
        
        # P4: Traffic (Purple) - Bottom Lane
        players.append({
            'start': (2.0, 3.0),
            'goal': (18.0, 3.0),
            'color': 'purple',
            'is_ambulance': False,
            'is_slow': False
        })
        
    else:
        # Uniform / Standard Ambulance setup
        # Start x=2. y spread 2.0 to 8.0
        y_starts = np.linspace(2.0, 8.0, n_agents)
        y_goals = np.linspace(2.0, 8.0, n_agents) 
        
        for i in range(n_agents):
            # Check if Ambulance
            is_ambulance = (mode == "Ambulance" and i == n_agents - 1) # Last agent is ambulance
            
            color = 'magenta' if is_ambulance else colors[i % len(colors)]
            
            players.append({
                'start': (2.0, y_starts[i]),
                'goal': (18.0, y_goals[i]),
                'color': color,
                'is_ambulance': is_ambulance
            })
        
    return players

# ==========================================
# GAME CREATION HELPER
# ==========================================

def create_bottleneck_game(n_agents, obstacles, tau=30, dt=0.2, mode="Uniform"):
    """
    Creates a car game with variable number of agents.
    mode: "Uniform" or "Ambulance"
    """
    players_config = get_players_config_funnel(n_agents, mode)
    
    # Dimensions
    d = 5
    m = 2
    
    # Cost matrices (Standard)
    Qhat = np.diag(np.array([50.0, 10.0, 5.0, 5.0, 2.0]))
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    Ri = np.diag(np.array([8.0, 4.0])) # Standard cost
    
    # Ambulance Costs (Higher priority on tracking)
    # Higher Qtau (Goal), Higher Qhat (Reference tracking)
    Qhat_amb = Qhat * 2.0 
    Qtau_amb = Qtau * 2.0
    
    # Dynamics
    def car_dynamics(x, u):
        p, q, theta, v, omega = x
        delta_v, delta_omega = u
        p_next = p + dt * v * np.cos(theta)
        q_next = q + dt * v * np.sin(theta)
        theta_next = theta + dt * omega
        v_next = v + delta_v
        omega_next = omega + delta_omega
        return np.array([p_next, q_next, theta_next, v_next, omega_next])
    
    def g_dummy(x, u):
        return np.array([0.0])
        
    players = []
    
    for cfg in players_config:
        start = cfg['start']
        goal = cfg['goal']
        is_ambulance = cfg.get('is_ambulance', False)
        is_slow = cfg.get('is_slow', False)
        
        xref = np.zeros((d, tau))
        t_range = np.arange(tau, dtype=float)
        
        # Improved Reference with Waypoint through bottleneck
        waypoint = np.array([10.0, 5.0])
        
        # Distance check for speed ref
        dist_total = np.linalg.norm(np.array(goal) - np.array(start))
        
        # Target Velocity
        duration = (tau - 1) * dt
        base_v = dist_total / duration
        
        if is_ambulance:
            target_v = base_v * 1.5 
        elif is_slow:
            # Slow car moves at 50% speed
            target_v = base_v * 0.5
        else:
            target_v = base_v
            
        # Recalculate path based on target_v?
        # If target_v is high, it reaches goal at t < tau.
        # We need to construct xref accordingly.
        
        t_arrival = dist_total / target_v if target_v > 0 else duration
        t_arrival_idx = int(t_arrival / dt)
        t_arrival_idx = min(t_arrival_idx, tau-1)
        
        # Waypoint timing
        t_mid_idx = t_arrival_idx // 2
        
        # Construct Xref
        # 1. Start -> Waypoint (0 to t_mid)
        for t in range(t_mid_idx):
            s = t / max(1, t_mid_idx)
            pos = np.array(start) + s * (waypoint - np.array(start))
            xref = xref.at[0, t].set(pos[0])
            xref = xref.at[1, t].set(pos[1])
            angle = np.arctan2(waypoint[1]-start[1], waypoint[0]-start[0])
            xref = xref.at[2, t].set(angle)
            xref = xref.at[3, t].set(target_v)
            
        # 2. Waypoint -> Goal (t_mid to t_arrival)
        for t in range(t_mid_idx, t_arrival_idx + 1):
            s = (t - t_mid_idx) / max(1, (t_arrival_idx - t_mid_idx))
            pos = waypoint + s * (np.array(goal) - waypoint)
            xref = xref.at[0, t].set(pos[0])
            xref = xref.at[1, t].set(pos[1])
            angle = np.arctan2(goal[1]-waypoint[1], goal[0]-waypoint[0])
            xref = xref.at[2, t].set(angle)
            xref = xref.at[3, t].set(target_v)
            
        # 3. Goal -> End (Stationary)
        for t in range(t_arrival_idx + 1, tau):
            xref = xref.at[0, t].set(goal[0])
            xref = xref.at[1, t].set(goal[1])
            xref = xref.at[2, t].set(0.0) # Arbitrary angle?
            xref = xref.at[3, t].set(0.0) # Stop
            
        # Assign Weights
        curr_Qi = Qi
        curr_Qtau = Qtau_amb if is_ambulance else Qtau
        # We can't change Qhat easily in Player init without refactoring?
        # Player takes Qi, Qtau, Ri.
        # Qhat is used to build Qi/Qtau usually.
        # Here we pass pre-built Qi/Qtau.
        # Let's boost Qi for ambulance too (tracking cost)
        if is_ambulance:
            curr_Qi = Qi * 2.0
            
        p = Player(xref=xref, f=car_dynamics, g=g_dummy, tau=tau, Qi=curr_Qi, Qtau=curr_Qtau, Ri=Ri)
        players.append(p)
        
    # Joint Constraints
    r_col = 1.0 # Relaxed to 1.0 from 1.2 to allow tighter packing in widened bottleneck
    u_bound = np.array([0.2, 0.8])
    
    def g_constraints(x_joint, u_joint):
        constraints = []
        
        # 1. Collision
        for i in range(n_agents):
            xi = x_joint[i*d:(i+1)*d]
            for j in range(i+1, n_agents):
                xj = x_joint[j*d:(j+1)*d]
                dist = np.sqrt((xi[0] - xj[0])**2 + (xi[1] - xj[1])**2)
                constraints.append(-dist + r_col)
                
        # 2. Obstacles
        for i in range(n_agents):
            xi = x_joint[i*d:(i+1)*d]
            for obs in obstacles:
                dist = np.sqrt((xi[0] - obs['x'])**2 + (xi[1] - obs['y'])**2)
                constraints.append(-dist + obs['r']) # obs['r'] includes car radius usually?
                # obs['r'] is obstacle physical radius. 
                # Distance should be > car_radius + obs_radius.
                # If we assume point mass car, just obs['r'].
                # But car has physical size.
                # Let's assume car radius ~ 0.5 (consistent with r_col=1.0)
                # So min dist = 0.5 + 0.5 = 1.0?
                # Current map.py uses r=0.7.
                # Let's stick to the formula: -dist + (obs_r + car_r) <= 0
                # Here we just use obs['r']. Let's increase obs['r'] effectively to account for car size.
                # Effective obstacle radius = 0.5 + 0.5 = 1.0.
                constraints.append(-dist + (obs['r'] + 0.5))
                
        # 3. Velocity
        for i in range(n_agents):
            xi = x_joint[i*d:(i+1)*d]
            constraints.append(-xi[3]) # v >= 0
            
        # 4. Control
        for i in range(n_agents):
            ui = u_joint[i*m:(i+1)*m]
            constraints.extend(ui - u_bound)
            constraints.extend(-ui - u_bound)
            
        if not constraints: return np.array([0.0])
        return np.array(constraints)
        
    game = PotientialGame(players=players, g=g_constraints)
    game.type = 'car'
    game.obstacles_list = obstacles
    return game, players_config

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(x_sol, u_sol, game, players_config, dt=0.2):
    n = game.n
    d = 5
    tau = game.tau
    
    metrics = {}
    
    # 1. Average Velocity (Flow)
    # Mean velocity across all agents and all time steps
    # Extract v channel (index 3)
    velocities = []
    for i in range(n):
        v_traj = x_sol[i*d + 3, :]
        velocities.extend(v_traj)
    metrics['avg_velocity'] = float(np.mean(np.array(velocities)))
    
    # 2. Total Detour % (Weaving)
    # Compare actual path length to straight line distance (Start -> Goal)
    total_detour = 0.0
    
    for i in range(n):
        # Actual Path Length
        traj = x_sol[i*d : i*d+2, :]
        # Sum of euclidean distances between steps
        diffs = np.diff(traj, axis=1)
        path_len = np.sum(np.sqrt(np.sum(diffs**2, axis=0)))
        
        # Straight Line
        start = players_config[i]['start']
        goal = players_config[i]['goal']
        straight_len = np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)
        
        if straight_len > 0:
            detour_pct = (path_len - straight_len) / straight_len * 100.0
            total_detour += detour_pct
            
    metrics['avg_detour_pct'] = float(total_detour / n)
    
    # 3. Congestion / Lane Formation (Heuristic)
    # Standard deviation of Y positions in the bottleneck (x=8 to x=12)
    # Low std dev => Single file line
    # High std dev => Multiple lanes / Disorganized
    
    y_positions_in_bottleneck = []
    for i in range(n):
        traj_x = x_sol[i*d, :]
        traj_y = x_sol[i*d + 1, :]
        
        # Filter points where 8 <= x <= 12
        mask = (traj_x >= 8.0) & (traj_x <= 12.0)
        if np.any(mask):
            y_positions_in_bottleneck.extend(traj_y[mask])
            
    if y_positions_in_bottleneck:
        metrics['bottleneck_y_std'] = float(np.std(np.array(y_positions_in_bottleneck)))
    else:
        metrics['bottleneck_y_std'] = 0.0
        
    # 4. Ambulance Metrics
    # Check if there is an ambulance in the game
    # config has 'is_ambulance' key
    has_ambulance = any([cfg.get('is_ambulance', False) for cfg in players_config])
    
    if has_ambulance:
        # Extract Ambulance velocity
        # Assuming only ONE ambulance for now
        for i, cfg in enumerate(players_config):
            if cfg.get('is_ambulance', False):
                # Get its avg velocity
                v_amb = x_sol[i*d + 3, :]
                metrics['ambulance_velocity'] = float(np.mean(v_amb))
                
                # Traffic velocity (everyone else)
                v_traffic = []
                for j in range(n):
                    if j != i:
                        v_traffic.extend(x_sol[j*d + 3, :])
                metrics['traffic_velocity'] = float(np.mean(np.array(v_traffic)))
                
    return metrics

# ==========================================
# PLOTTING
# ==========================================

def plot_map(obstacles, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAP_X_LIMIT)
    ax.set_ylim(0, MAP_Y_LIMIT)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Draw Obstacles
    for obs in obstacles:
        # Draw effective radius (inner) and clearance radius (outer)?
        # Just draw physical
        c = plt.Circle((obs['x'], obs['y']), obs['r'], color='black')
        ax.add_patch(c)
        
    # Draw Funnel lines
    plt.axvline(x=8, color='r', linestyle=':', alpha=0.3)
    plt.axvline(x=12, color='r', linestyle=':', alpha=0.3)
    
    ax.set_title("Experiment 2 Setup: Bottleneck")
    plt.savefig(filename)
    plt.close()

def save_raw_particles(name, game, trajs_np, output_dir):
    # trajs_np shape: (num_particles, tau, n_aug)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAP_X_LIMIT)
    ax.set_ylim(0, MAP_Y_LIMIT)
    ax.set_aspect('equal')
    
    # Obstacles
    for obs in game.obstacles_list:
        c = plt.Circle((obs['x'], obs['y']), obs['r'], color='black', alpha=0.3, linestyle=':')
        ax.add_patch(c)

    d = game.Qtau.shape[0] // game.n
    n_players = game.n
    colors = ['b', 'r', 'g', 'm', 'c']
    
    # Plot particles
    for i in range(len(trajs_np)):
        traj = trajs_np[i]
        # x part is first N*d cols
        x_traj = traj[:, :game.Qtau.shape[0]].T # (full_d, tau)
        
        for p in range(n_players):
            xi = x_traj[p*d:(p+1)*d, :]
            c = colors[p % len(colors)]
            ax.plot(xi[0, :], xi[1, :], color=c, alpha=0.05)

    ax.set_title(f"Raw Particles: {name}")
    plt.savefig(os.path.join(output_dir, f"experiment2_pf_{name}_raw_particles.png"))
    plt.close()

def save_mode_summary(name, game, modes_data, players_config, output_dir):
    # modes_data list of (x_sol, u_sol, cost, status)
    if not modes_data: return
    
    num_plots = len(modes_data)
    cols = min(num_plots, 2) # Max 2 cols
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, data in enumerate(modes_data):
        x_sol, u_sol, cost, status = data
        ax = axes[i] if i < len(axes) else None
        if ax is None: continue
        
        ax.set_xlim(0, MAP_X_LIMIT)
        ax.set_ylim(0, MAP_Y_LIMIT)
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Obstacles
        for obs in game.obstacles_list:
            c = plt.Circle((obs['x'], obs['y']), obs['r'], color='black', alpha=0.2)
            ax.add_patch(c)
            
        # Trajectories
        n = game.n
        d = 5
        for p in range(n):
            color = players_config[p]['color']
            traj = x_sol[p*d : p*d+2, :]
            ax.plot(traj[0, :], traj[1, :], '.-', color=color, alpha=0.8)
            ax.plot(traj[0, 0], traj[1, 0], 'o', color=color)
            ax.plot(traj[0, -1], traj[1, -1], 'x', color=color)

        ax.set_title(f"Mode {i+1}\n{status}, Cost: {cost:.2f}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"experiment2_pf_{name}_modes_summary.png"))
    plt.close()

def save_results(name, mode_idx, game, x_sol, players_config, output_dir):
    n = game.n
    d = 5
    tau = game.tau
    
    # 1. Trajectory Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAP_X_LIMIT)
    ax.set_ylim(0, MAP_Y_LIMIT)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Obstacles
    for obs in game.obstacles_list:
        c = plt.Circle((obs['x'], obs['y']), obs['r'], color='black', alpha=0.3)
        ax.add_patch(c)
        
    for i in range(n):
        color = players_config[i]['color']
        traj = x_sol[i*d : i*d+2, :]
        ax.plot(traj[0, :], traj[1, :], '.-', color=color, label=f"P{i+1}", alpha=0.7)
        
    ax.set_title(f"Flow: {name} (Mode {mode_idx})")
    plt.savefig(os.path.join(output_dir, f"experiment2_pf_{name}_mode{mode_idx}_traj.png"))
    plt.close()
    
    # 2. Animation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAP_X_LIMIT)
    ax.set_ylim(0, MAP_Y_LIMIT)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Obstacles static
    for obs in game.obstacles_list:
        c = plt.Circle((obs['x'], obs['y']), obs['r'], color='black', alpha=0.3)
        ax.add_patch(c)
        
    lines = [ax.plot([], [], 'o-', color=players_config[i]['color'])[0] for i in range(n)]
    
    def update(frame):
        for i in range(n):
            traj = x_sol[i*d : i*d+2, :frame+1]
            lines[i].set_data(traj[0], traj[1])
        return lines
        
    ani = FuncAnimation(fig, update, frames=tau, blit=True)
    ani.save(os.path.join(output_dir, f"experiment2_pf_{name}_mode{mode_idx}.gif"), writer='pillow', fps=10)
    plt.close()

# ==========================================
# MAIN
# ==========================================

def main():
    # Setup Artifacts
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiment2_pf")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Plot Map
    plot_map(OBSTACLES_SETUP, os.path.join(artifacts_dir, "experiment2_map.png"))
    
    results = []
    
    for name, n_agents in DENSITIES.items():
        print(f"\nRunning Density: {name} ({n_agents} agents) with PF...")
        
        # Determine Mode
        mode = name
        if name == "Blocked_Ambulance":
             mode = "Blocked_Ambulance"
        elif name == "Ambulance":
             mode = "Ambulance"
        else:
             mode = "Uniform"
        
        # Reduce tau to help solver, increase dt slightly
        tau = 30
        dt = 0.25
        
        game, config = create_bottleneck_game(n_agents, OBSTACLES_SETUP, tau=tau, dt=dt, mode=mode)
        
        # 1. Run Particle Filter to find modes
        print("  Running Particle Filter...")
        estimates, trajs_np, dist_matrix = get_coarse_estimates(game, num_particles=150, noise_scale=0.2, cluster_threshold=2.0)
        
        # Save Raw Particles Plot
        save_raw_particles(name, game, trajs_np, artifacts_dir)
        
        # Auto-tune eps logic (simplified from test_pf.py)
        eps_values = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        best_labels = None
        max_clusters = 0
        best_eps = 5.0
        
        for eps_test in eps_values:
            labels = dbscan_from_dist_matrix(dist_matrix, 150, eps=eps_test, min_samples=3)
            unique_l = np.unique(labels)
            n_clusters = len(unique_l[unique_l != -1])
            if n_clusters >= max_clusters and n_clusters > 0:
                max_clusters = n_clusters
                best_labels = labels
                best_eps = eps_test
        
        if max_clusters > 0:
            print(f"  Found {max_clusters} clusters with eps={best_eps}")
            # Extract estimates
            estimates = []
            unique_labels = np.unique(best_labels)
            valid_clusters = unique_labels[unique_labels != -1]
            full_d = game.Qtau.shape[0]
            
            for lab in valid_clusters:
                indices = np.where(best_labels == lab)[0]
                mean_traj = np.mean(trajs_np[indices], axis=0)
                x_sol = mean_traj[:, :full_d].T 
                u_sol = mean_traj[:, full_d:].T 
                estimates.append((x_sol, u_sol))
        else:
            print("  No clusters found, using single estimate from mean of all.")
            # If max_clusters == 0, implies everything is noise?
            # Just take global mean
            mean_traj = np.mean(trajs_np, axis=0)
            x_sol = mean_traj[:, :full_d].T
            u_sol = mean_traj[:, full_d:].T
            estimates = [(x_sol, u_sol)]

        print(f"  Refining {len(estimates)} modes...")
        
        current_modes_data = []
        
        for i, (x_est, u_est) in enumerate(estimates):
            print(f"    Mode {i+1}/{len(estimates)}...")
            
            # Solve with warm start
            try:
                x_sol, u_sol, res = solve_game(game, warm_start=(x_est, u_est))
                
                status = "Success" if res.success else "Failed"
                if hasattr(res, 'message') and "acceptable" in str(res.message).lower():
                    status = "Acceptable"
                
                print(f"      Result: {status}, Cost: {res.fun}")
                
                # Only save valid solutions? Or save all to see failures?
                if res.success or status == "Acceptable":
                     # Metrics
                    met = calculate_metrics(x_sol, u_sol, game, config, dt)
                    met['condition'] = name
                    met['n_agents'] = n_agents
                    met['mode_id'] = i
                    met['cost'] = res.fun
                    met['status'] = status
                    results.append(met)
                    
                    # Store for summary
                    current_modes_data.append((x_sol, u_sol, res.fun, status))
                    
                    # Save Artifacts
                    save_results(name, i, game, x_sol, config, artifacts_dir)
                else:
                    print("      Skipping artifacts for failed solution.")
                    
            except Exception as e:
                print(f"      Error solving mode {i+1}: {e}")
                
        # Save Mode Summary Plot
        if current_modes_data:
            save_mode_summary(name, game, current_modes_data, config, artifacts_dir)

    # CSV
    if results:
        df = pd.DataFrame(results)
        cols = ['condition', 'mode_id', 'status', 'cost', 'n_agents', 'avg_velocity', 'avg_detour_pct', 'bottleneck_y_std', 'ambulance_velocity', 'traffic_velocity']
        # Ensure columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        df.to_csv(os.path.join(artifacts_dir, "experiment2_pf.csv"), index=False)
        print(f"\nExperiment 2 PF Completed. Results at {artifacts_dir}")
    else:
        print("\nNo valid results found.")

if __name__ == "__main__":
    main()
