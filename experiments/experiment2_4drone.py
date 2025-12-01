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
# EXPERIMENT 2 (DRONE) SETUP: The Bottleneck
# ==========================================

# Map Dimensions
MAP_X_LIMIT = 5.0
MAP_Y_LIMIT = 5.0
MAP_Z_LIMIT = 3.0

# Drone size assumption ~0.15 radius
# We want a bottleneck at x=0.
# Drones start at x = -2.0 (Left) and go to x = 2.0 (Right).
# We use two spheres to create a gap in the middle (at origin).

def create_bottleneck_obstacles(gap_width, sphere_radius=0.5):
    # Use smaller spheres as pillars instead of massive walls
    # Gap is the distance between the surfaces of the spheres at z=0.5
    # y_offset = sphere_radius + gap_width / 2
    
    r = sphere_radius
    y_offset = r + gap_width / 2.0
    
    obstacles = []
    # Obstacle 1 (Positive Y pillar)
    obstacles.append({'x': 0.0, 'y': y_offset, 'z': 0.5, 'r': r})
    # Obstacle 2 (Negative Y pillar)
    obstacles.append({'x': 0.0, 'y': -y_offset, 'z': 0.5, 'r': r})
    
    return obstacles

SCENARIOS = {
    "Nominal": {
        "gap": 1.2, 
        "n_agents": 4,
        "mode": "Uniform"
    },
    "High": {
        "gap": 1.5, 
        "n_agents": 4,
        "mode": "Uniform"
    },
    "Very_Cramped": {
        "gap": 0.8, 
        "n_agents": 4,
        "mode": "Uniform"
    },
    "Blocked_Ambulance_Nominal": {
        "gap": 1.2, 
        "n_agents": 4,
        "mode": "Blocked_Ambulance"
    },
    "Blocked_Ambulance_Cramped": {
        "gap": 0.8, 
        "n_agents": 4,
        "mode": "Blocked_Ambulance"
    }
}

def get_players_config(n_agents, mode="Uniform"):
    players = []
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Start/Goal X
    start_x = -2.0
    goal_x = 2.0
    flight_z = 0.5
    
    if mode == "Blocked_Ambulance":
        # 4 Agents
        # 1. Slow drone (Blue) - Center-ish
        players.append({
            'start': (start_x + 0.5, 0.0, flight_z), # Slightly ahead
            'goal': (goal_x, 0.0, flight_z),
            'color': 'blue',
            'is_ambulance': False,
            'is_slow': True
        })
        
        # 2. Ambulance (Magenta) - Behind Slow
        players.append({
            'start': (start_x, 0.0, flight_z),
            'goal': (goal_x, 0.0, flight_z),
            'color': 'magenta',
            'is_ambulance': True,
            'is_slow': False
        })
        
        # 3. Traffic (Green) - Side
        players.append({
            'start': (start_x + 0.2, 0.3, flight_z),
            'goal': (goal_x, 0.3, flight_z),
            'color': 'green',
            'is_ambulance': False,
            'is_slow': False
        })
        
        # 4. Traffic (Purple) - Side
        players.append({
            'start': (start_x + 0.2, -0.3, flight_z),
            'goal': (goal_x, -0.3, flight_z),
            'color': 'purple',
            'is_ambulance': False,
            'is_slow': False
        })
        
    else:
        # Uniform
        # Start in a generic line along Y
        y_positions = np.linspace(-0.4, 0.4, n_agents)
        
        for i in range(n_agents):
            players.append({
                'start': (start_x, y_positions[i], flight_z),
                'goal': (goal_x, y_positions[i], flight_z), # Aim for same Y, forcing squeeze
                'color': colors[i % len(colors)],
                'is_ambulance': False,
                'is_slow': False
            })
            
    return players

# ==========================================
# GAME CREATION
# ==========================================

def create_bottleneck_game(scenario_name, config, tau=40, dt=0.1):
    gap = config['gap']
    n_agents = config['n_agents']
    mode = config['mode']
    
    obstacles = create_bottleneck_obstacles(gap)
    players_config = get_players_config(n_agents, mode)
    
    # Dimensions
    d = 8
    m = 3
    
    # Cost Matrices
    # Standard
    diag_pos = np.array([50.0, 50.0, 50.0])
    diag_ang = np.array([10.0, 50.0]) # High phi cost
    diag_v   = np.array([5.0])
    diag_rate= np.array([20.0, 2.0]) # High yaw rate cost
    
    Qhat = np.diag(np.concatenate([diag_pos, diag_ang, diag_v, diag_rate]))
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    Ri = np.diag(np.array([8.0, 4.0, 4.0])) # Control cost
    
    # Dynamics
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
        return np.array([p_next, q_next, r_next, theta_next, phi_next, v_next, omega_theta_next, omega_phi_next])
    
    def g_dummy(x, u): return np.array([0.0])
    
    players = []
    for cfg in players_config:
        start = cfg['start']
        goal = cfg['goal']
        is_ambulance = cfg.get('is_ambulance', False)
        is_slow = cfg.get('is_slow', False)
        
        # Weights customization
        curr_Qi = Qi
        curr_Qtau = Qtau
        curr_Ri = Ri
        
        if is_ambulance:
            curr_Qi = Qi * 2.0
            curr_Qtau = Qtau * 2.0 # Urgency
            
        # Reference Trajectory
        xref = np.zeros((d, tau))
        
        # Waypoint at bottleneck (0, 0, 0.5) to encourage going through gap
        # Otherwise they might go around the massive spheres if they are too large?
        # Gap is at x=0.
        waypoint = np.array([0.0, 0.0, 0.5])
        
        dist_total = np.linalg.norm(np.array(goal) - np.array(start))
        duration = (tau - 1) * dt
        base_v = dist_total / duration
        
        if is_ambulance: target_v = base_v * 1.5
        elif is_slow: target_v = base_v * 0.5
        else: target_v = base_v
        
        t_range = np.arange(tau, dtype=float)
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dz = goal[2] - start[2]
        
        # Angles: Align yaw (theta) with direction of travel
        ref_theta = np.arctan2(dy, dx)
        ref_phi = 0.0
        
        # Time to arrival
        if target_v > 0:
            t_arrival = dist_total / target_v
            t_idx = int(min(t_arrival/dt, tau-1))
        else:
            t_idx = tau-1
            
        # Fill xref
        for t in range(tau):
            if t <= t_idx:
                s = t / max(1, t_idx)
                xref = xref.at[0, t].set(start[0] + s*dx)
                xref = xref.at[1, t].set(start[1] + s*dy)
                xref = xref.at[2, t].set(start[2] + s*dz)
                xref = xref.at[3, t].set(ref_theta)
                xref = xref.at[4, t].set(ref_phi)
                xref = xref.at[5, t].set(target_v)
            else:
                # Stay at goal
                xref = xref.at[0, t].set(goal[0])
                xref = xref.at[1, t].set(goal[1])
                xref = xref.at[2, t].set(goal[2])
                xref = xref.at[3, t].set(ref_theta)
                xref = xref.at[4, t].set(ref_phi)
                xref = xref.at[5, t].set(0.0)
                
        p = Player(xref=xref, f=drone_dynamics_local, g=g_dummy, tau=tau, Qi=curr_Qi, Qtau=curr_Qtau, Ri=curr_Ri)
        players.append(p)
        
    # Joint Constraints
    r_col = 0.2 # Drone radius + buffer
    u_bound = np.array([0.15, 0.75, 0.75])
    
    def g_constraints(x_joint, u_joint):
        constraints = []
        
        # 1. Collision
        for i in range(n_agents):
            xi = x_joint[i*d:(i+1)*d]
            for j in range(i+1, n_agents):
                xj = x_joint[j*d:(j+1)*d]
                dist = np.sqrt((xi[0]-xj[0])**2 + (xi[1]-xj[1])**2 + (xi[2]-xj[2])**2)
                constraints.append(-dist + r_col)
                
        # 2. Obstacles (Spheres)
        for i in range(n_agents):
            xi = x_joint[i*d:(i+1)*d]
            for obs in obstacles:
                dist = np.sqrt((xi[0]-obs['x'])**2 + (xi[1]-obs['y'])**2 + (xi[2]-obs['z'])**2)
                # obs['r'] is the sphere radius. Drone has radius ~0.1.
                # constraint: dist > obs_r + drone_r
                constraints.append(-dist + (obs['r'] + 0.1))
                
        # 3. Positive velocity
        for i in range(n_agents):
            xi = x_joint[i*d:(i+1)*d]
            constraints.append(-xi[5])
            
        # 4. Control bounds
        for i in range(n_agents):
            ui = u_joint[i*m:(i+1)*m]
            constraints.extend(ui - u_bound)
            constraints.extend(-ui - u_bound)
            
        if not constraints: return np.array([0.0])
        return np.array(constraints)
        
    game = PotientialGame(players=players, g=g_constraints)
    game.type = 'drone'
    game.obstacles_list = obstacles
    return game, players_config

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(x_sol, game, players_config, dt=0.1):
    n = game.n
    d = 8
    tau = game.tau
    
    metrics = {}
    
    # 1. Avg Velocity
    velocities = []
    for i in range(n):
        velocities.extend(x_sol[i*d + 5, :]) # v index is 5
    metrics['avg_velocity'] = float(np.mean(np.array(velocities)))
    
    # 2. Congestion (Min pairwise distance in bottleneck region)
    # Region: x in [-0.5, 0.5]
    min_dists = []
    for t in range(tau):
        # Check if any agent is in bottleneck
        in_bottleneck = False
        positions = []
        for i in range(n):
            px = x_sol[i*d, t]
            if -0.5 <= px <= 0.5:
                in_bottleneck = True
            positions.append(x_sol[i*d : i*d+3, t])
            
        if in_bottleneck:
            # Calc pairwise dists at this step
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_dists.append(dist)
                    
    if min_dists:
        metrics['min_dist_bottleneck'] = float(np.min(min_dists))
    else:
        metrics['min_dist_bottleneck'] = 0.0 # No one entered?
        
    # 3. Ambulance Velocity vs Traffic
    has_ambulance = any([cfg.get('is_ambulance', False) for cfg in players_config])
    if has_ambulance:
        amb_idx = [i for i, cfg in enumerate(players_config) if cfg.get('is_ambulance')][0]
        v_amb = x_sol[amb_idx*d + 5, :]
        metrics['ambulance_velocity'] = float(np.mean(v_amb))
        
        traffic_vs = []
        for i in range(n):
            if i != amb_idx:
                traffic_vs.extend(x_sol[i*d + 5, :])
        metrics['traffic_velocity'] = float(np.mean(np.array(traffic_vs)))
        
    return metrics

# ==========================================
# PLOTTING & SAVING
# ==========================================

def plot_map_setup(obstacles, players_config, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 2)
    
    # Draw Obstacles (Spheres)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for obs in obstacles:
        x = obs['x'] + obs['r'] * np.cos(u) * np.sin(v)
        y = obs['y'] + obs['r'] * np.sin(u) * np.sin(v)
        z = obs['z'] + obs['r'] * np.cos(v)
        ax.plot_wireframe(x, y, z, color="black", alpha=0.2)
        
    # Draw Players Start/Goal
    for i, cfg in enumerate(players_config):
        s = cfg['start']
        g = cfg['goal']
        c = cfg['color']
        ax.scatter(s[0], s[1], s[2], color=c, marker='o', label=f"Start {i}")
        ax.scatter(g[0], g[1], g[2], color=c, marker='x', label=f"Goal {i}")
        
    plt.title("Experiment 2 (Drone) Setup: Bottleneck")
    plt.savefig(filename)
    plt.close()

def save_results(name, game, x_sol, players_config, output_dir):
    n = game.n
    d = 8
    tau = game.tau
    
    # 1. Static Trajectory Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 2)
    
    # Obstacles
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for obs in game.obstacles_list:
        x = obs['x'] + obs['r'] * np.cos(u) * np.sin(v)
        y = obs['y'] + obs['r'] * np.sin(u) * np.sin(v)
        z = obs['z'] + obs['r'] * np.cos(v)
        ax.plot_wireframe(x, y, z, color="black", alpha=0.1)
        
    for i in range(n):
        traj = x_sol[i*d : i*d+3, :]
        c = players_config[i]['color']
        ax.plot(traj[0, :], traj[1, :], traj[2, :], '.-', color=c)
        
    plt.title(f"Trajectories - {name}")
    plt.savefig(os.path.join(output_dir, f"experiment2_4drone_traj_{name}.png"))
    plt.close()
    
    # 2. Animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 2)
    
    # Static obstacles
    for obs in game.obstacles_list:
        x = obs['x'] + obs['r'] * np.cos(u) * np.sin(v)
        y = obs['y'] + obs['r'] * np.sin(u) * np.sin(v)
        z = obs['z'] + obs['r'] * np.cos(v)
        ax.plot_wireframe(x, y, z, color="black", alpha=0.1)
        
    lines = [ax.plot([], [], [], 'o-', color=players_config[i]['color'])[0] for i in range(n)]
    
    def update(frame):
        for i in range(n):
            traj = x_sol[i*d : i*d+3, :frame+1]
            lines[i].set_data(traj[0], traj[1])
            lines[i].set_3d_properties(traj[2])
        return lines
        
    ani = FuncAnimation(fig, update, frames=tau, blit=True)
    ani.save(os.path.join(output_dir, f"experiment2_4drone_{name}.gif"), writer='pillow', fps=10)
    plt.close()

def save_detailed_csvs(name, x_sol, dt, output_dir):
    # Format: drone_id,t,x,y,z,yaw
    n = 4
    d = 8
    tau = x_sol.shape[1]
    
    data = []
    for i in range(n):
        drone_id = f"cf{i+1}"
        for t in range(tau):
            time = t * dt
            idx = i * d
            data.append({
                'drone_id': drone_id,
                't': time,
                'x': x_sol[idx+0, t],
                'y': x_sol[idx+1, t],
                'z': x_sol[idx+2, t],
                'yaw': x_sol[idx+3, t]
            })
            
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, f"drone_exp2_x_{name}.csv"), index=False, float_format='%.4f')


# ==========================================
# MAIN
# ==========================================

def run_single_scenario(name, config, artifacts_dir):
    print(f"\nRunning Scenario: {name}")
    
    dt_val = 0.1
    game, players_config = create_bottleneck_game(name, config, tau=40, dt=dt_val)
    
    # Solve
    seed = 42
    x_sol, u_sol, res = solve_game(game, seed=seed)
    
    if not res.success and "acceptable" not in str(res.message).lower():
        print(f"Warning: Solver did not converge for {name}")
        
    # Metrics
    met = calculate_metrics(x_sol, game, players_config, dt=dt_val)
    met['condition'] = name
    
    print(f"Metrics: {met}")
    
    # Save Artifacts
    save_results(name, game, x_sol, players_config, artifacts_dir)
    save_detailed_csvs(name, x_sol, dt_val, artifacts_dir)
    
    # Save partial metrics for parallel execution
    res_df = pd.DataFrame([met])
    res_file = os.path.join(artifacts_dir, f"metrics_{name}.csv")
    res_df.to_csv(res_file, index=False)
    
    return met

def main():
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiment2_4drone")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Setup sample map
    # Use Nominal for map plot
    temp_obs = create_bottleneck_obstacles(SCENARIOS["Nominal"]["gap"])
    temp_cfg = get_players_config(4, "Uniform")
    plot_map_setup(temp_obs, temp_cfg, os.path.join(artifacts_dir, "experiment2_4drone_map.png"))
    
    # Check for command line argument
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        if scenario_name in SCENARIOS:
            run_single_scenario(scenario_name, SCENARIOS[scenario_name], artifacts_dir)
            return
        else:
             print(f"Scenario {scenario_name} not found. Running all.")
    
    results = []
    
    for name, config in SCENARIOS.items():
        met = run_single_scenario(name, config, artifacts_dir)
        results.append(met)
        
    # Summary CSV
    df = pd.DataFrame(results)
    cols = ['condition', 'avg_velocity', 'min_dist_bottleneck', 'ambulance_velocity', 'traffic_velocity']
    df = df[cols]
    df.to_csv(os.path.join(artifacts_dir, "experiment2_4drone.csv"), index=False)
    print(f"\nExperiment 2 (Drone) Completed.")

if __name__ == "__main__":
    main()
