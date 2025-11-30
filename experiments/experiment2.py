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
# EXPERIMENT 2 SETUP: The Bottleneck Capacity
# ==========================================

# Map Dimensions
MAP_X_LIMIT = 20.0
MAP_Y_LIMIT = 10.0
CORRIDOR_Y_MIN = 4.0
CORRIDOR_Y_MAX = 6.0

# Define the bottleneck obstacles
# A narrow corridor in the middle: width = 2.0 (y=4 to y=6)
# Walls above y=6 and below y=4 from x=8 to x=12?
# Or just continuous walls with a gap?
# Let's creating a "funnel" or just a simple narrow passage.
# Simple narrow passage from x=5 to x=15.

def create_bottleneck_obstacles():
    obstacles = []
    # Top Wall (y > 6)
    # Create a block of obstacles
    for x in np.linspace(0, 20, 21): # Every 1 unit
        for y in np.linspace(6.5, 10, 5):
             obstacles.append({'x': float(x), 'y': float(y), 'r': 0.5})
             
    # Bottom Wall (y < 4)
    for x in np.linspace(0, 20, 21):
        for y in np.linspace(0, 3.5, 5):
             obstacles.append({'x': float(x), 'y': float(y), 'r': 0.5})
             
    return obstacles

OBSTACLES_SETUP = create_bottleneck_obstacles()

# Variable: Number of Agents (Density)
# All agents start on left (x=2) and want to go right (x=18)
# Random y start within safe zone (y=4-6) or slightly spread out?
# Let's spread them out at x=2, y=[3, 7] but they have to squeeze through y=[4,6]

DENSITIES = {
    "2_Agents": 2,
    "3_Agents": 3,
    "4_Agents": 4
}

def get_players_config(n_agents):
    players = []
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Start positions: clustered around x=2, distributed in y
    # Goal positions: x=18, same y or clustered?
    
    # Let's fan them out at start to force convergence
    y_starts = np.linspace(2.0, 8.0, n_agents) # Wide start
    
    # But obstacles are at y<4 and y>6. 
    # So if they start at y=2 or y=8, they are INSIDE obstacles.
    # They must start in the "safe" zone or the "funnel" must be open at ends.
    # Let's make the obstacles only present in the middle section x=8 to x=12.
    # That creates a true "Bottleneck".
    
    return players

# Redefine obstacles to be a middle bottleneck
def create_funnel_obstacles():
    obstacles = []
    # Two massive obstacles to create a gap between y=3.5 and y=6.5 at x=10
    # Gap = 3.0m
    # Top Pillar (Center (10, 9.0), Radius 2.5) -> Bottom edge at y=6.5
    obstacles.append({'x': 10.0, 'y': 9.0, 'r': 2.5})
    
    # Bottom Pillar (Center (10, 1.0), Radius 2.5) -> Top edge at y=3.5
    obstacles.append({'x': 10.0, 'y': 1.0, 'r': 2.5})
    
    return obstacles

OBSTACLES_SETUP = create_funnel_obstacles()

def get_players_config_funnel(n_agents):
    players = []
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Start x=2. y spread 2.0 to 8.0 (Safe because obstacles are only at x=8..12)
    y_starts = np.linspace(2.0, 8.0, n_agents)
    y_goals = np.linspace(2.0, 8.0, n_agents) # Fan out again?
    # Or everyone wants to go to y=5? That creates more conflict.
    # Let's have them try to maintain their y, but forced to squeeze.
    
    for i in range(n_agents):
        players.append({
            'start': (2.0, y_starts[i]),
            'goal': (18.0, y_goals[i]),
            'color': colors[i % len(colors)]
        })
        
    return players

# ==========================================
# GAME CREATION HELPER
# ==========================================

def create_bottleneck_game(n_agents, obstacles, tau=30, dt=0.2):
    """
    Creates a car game with variable number of agents.
    """
    players_config = get_players_config_funnel(n_agents)
    
    # Dimensions
    d = 5
    m = 2
    
    # Cost matrices (Standard)
    Qhat = np.diag(np.array([50.0, 10.0, 5.0, 5.0, 2.0]))
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    Ri = np.diag(np.array([8.0, 4.0])) # Standard cost
    
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
        
        xref = np.zeros((d, tau))
        t_range = np.arange(tau, dtype=float)
        
        # Improved Reference with Waypoint through bottleneck
        # Waypoint: (10.0, 5.0) - Center of bottleneck
        
        # Define segments
        # Start -> Waypoint -> Goal
        waypoint = np.array([10.0, 5.0])
        
        # Segment 1: Start to Waypoint (First half of time)
        t_mid = tau // 2
        
        # Distance check for speed ref
        dist_total = np.linalg.norm(np.array(goal) - np.array(start))
        ref_speed = dist_total / ((tau-1)*dt)

        # Seg 1 (0 to t_mid)
        for t in range(t_mid):
            s = t / t_mid
            # Linear interp position
            pos = np.array(start) + s * (waypoint - np.array(start))
            xref = xref.at[0, t].set(pos[0])
            xref = xref.at[1, t].set(pos[1])
            # Angle
            angle = np.arctan2(waypoint[1]-start[1], waypoint[0]-start[0])
            xref = xref.at[2, t].set(angle)
            xref = xref.at[3, t].set(ref_speed) # Approx const speed
            
        # Seg 2 (t_mid to tau)
        for t in range(t_mid, tau):
            s = (t - t_mid) / (tau - 1 - t_mid)
            pos = waypoint + s * (np.array(goal) - waypoint)
            xref = xref.at[0, t].set(pos[0])
            xref = xref.at[1, t].set(pos[1])
            angle = np.arctan2(goal[1]-waypoint[1], goal[0]-waypoint[0])
            xref = xref.at[2, t].set(angle)
            xref = xref.at[3, t].set(ref_speed)
            
        p = Player(xref=xref, f=car_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
        players.append(p)
        
    # Joint Constraints
    r_col = 1.0 # Reduced from 1.2 to allow tighter squeezing
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
    metrics['avg_velocity'] = float(np.mean(velocities))
    
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
        metrics['bottleneck_y_std'] = float(np.std(y_positions_in_bottleneck))
    else:
        metrics['bottleneck_y_std'] = 0.0
        
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

def save_results(name, game, x_sol, players_config, output_dir):
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
        
    ax.set_title(f"Flow: {name}")
    plt.savefig(os.path.join(output_dir, f"experiment2_traj_{name}.png"))
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
    ani.save(os.path.join(output_dir, f"experiment2_{name}.gif"), writer='pillow', fps=10)
    plt.close()

# ==========================================
# MAIN
# ==========================================

def main():
    # Setup Artifacts
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiment2")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Plot Map
    plot_map(OBSTACLES_SETUP, os.path.join(artifacts_dir, "experiment2_map.png"))
    
    results = []
    
    for name, n_agents in DENSITIES.items():
        print(f"\nRunning Density: {name} ({n_agents} agents)")
        
        # Reduce tau to help solver, increase dt slightly
        tau = 30
        dt = 0.25
        
        game, config = create_bottleneck_game(n_agents, OBSTACLES_SETUP, tau=tau, dt=dt)
        
        # Solve
        seed = 42
        x_sol, u_sol, res = solve_game(game, seed=seed)
        
        if not res.success and "acceptable" not in str(res.message).lower():
            print(f"Warning: Solver did not converge for {name}")
            
        # Metrics
        met = calculate_metrics(x_sol, u_sol, game, config, dt)
        met['condition'] = name
        met['n_agents'] = n_agents
        results.append(met)
        
        print(f"Metrics: {met}")
        
        # Save
        save_results(name, game, x_sol, config, artifacts_dir)
        
    # CSV
    df = pd.DataFrame(results)
    cols = ['condition', 'n_agents', 'avg_velocity', 'avg_detour_pct', 'bottleneck_y_std']
    df = df[cols]
    df.to_csv(os.path.join(artifacts_dir, "experiment2.csv"), index=False)
    print(f"\nExperiment 2 Completed. Results at {artifacts_dir}")

if __name__ == "__main__":
    main()

