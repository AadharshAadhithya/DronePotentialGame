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
# EXPERIMENT 3 SETUP: The Volumetric Cost of Safety
# ==========================================

# Scenario: "The Converging 4"
# 4 Drones at N, S, E, W. Distance R=10 from center (0,0). Altitude Z=5.
# Converging to (0,0,5).
# Then continuing to opposite side.

RADIUS = 10.0
ALTITUDE = 5.0
SPEED = 5.0 # m/s? Distance 20m. 
# If tau=30, dt=0.1 -> 3s total. v = 20/3 = 6.6 m/s.
# Let's say tau=40, dt=0.1 -> 4s. v = 20/4 = 5.0 m/s.

R_COL_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]

def get_converging_players(r_col):
    # Start -> Goal
    # 1. North -> South: (0, 10, 5) -> (0, -10, 5)
    # 2. South -> North: (0, -10, 5) -> (0, 10, 5)
    # 3. East -> West: (10, 0, 5) -> (-10, 0, 5)
    # 4. West -> East: (-10, 0, 5) -> (10, 0, 5)
    
    configs = [
        {'start': (0.0, 10.0, 5.0), 'goal': (0.0, -10.0, 5.0), 'color': 'blue'},
        {'start': (0.0, -10.0, 5.0), 'goal': (0.0, 10.0, 5.0), 'color': 'red'},
        {'start': (10.0, 0.0, 5.0), 'goal': (-10.0, 0.0, 5.0), 'color': 'green'},
        {'start': (-10.0, 0.0, 5.0), 'goal': (10.0, 0.0, 5.0), 'color': 'purple'}
    ]
    return configs

def create_converging_drone_game(r_col, tau=40, dt=0.1):
    configs = get_converging_players(r_col)
    n_players = len(configs)
    
    # Drone Dimensions
    d = 8 # [p, q, r, theta, phi, v, omega_theta, omega_phi]
    m = 3 # [delta_v, delta_omega_theta, delta_omega_phi]
    
    # Cost Matrices
    # Weights: pos(3), ang(2), v(1), rates(2)
    diag_entries = np.array([50.0, 50.0, 50.0, 
                            10.0, 10.0, 
                            5.0, 
                            5.0, 2.0])
    Qhat = np.diag(diag_entries)
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    Ri = np.diag(np.array([8.0, 4.0, 4.0]))
    
    # Dynamics
    def drone_dynamics(x, u):
        p, q, r, theta, phi, v, omega_theta, omega_phi = x
        delta_v, delta_omega_theta, delta_omega_phi = u
        
        p_next = p + dt * v * np.cos(theta) * np.cos(phi)
        q_next = q + dt * v * np.sin(theta) * np.cos(phi)
        r_next = r + dt * v * np.sin(phi)
        
        theta_next = theta + dt * omega_theta
        phi_next = phi + dt * omega_phi
        
        v_next = v + delta_v
        omega_theta_next = omega_theta + delta_omega_theta
        omega_phi_next = omega_phi + delta_omega_phi
        
        return np.array([p_next, q_next, r_next, theta_next, phi_next, v_next, omega_theta_next, omega_phi_next])
        
    def g_dummy(x, u):
        return np.array([0.0])
        
    players = []
    
    for cfg in configs:
        start = np.array(cfg['start'])
        goal = np.array(cfg['goal'])
        
        xref = np.zeros((d, tau))
        t_range = np.arange(tau, dtype=float)
        
        # Linear interpolation
        dist_vec = goal - start
        dist = np.linalg.norm(dist_vec)
        
        # Angles
        # theta = atan2(dy, dx)
        # phi = atan2(dz, sqrt(dx^2+dy^2)) -> dz=0 so phi=0
        theta_ref = np.arctan2(dist_vec[1], dist_vec[0])
        phi_ref = 0.0
        
        v_ref = dist / ((tau-1)*dt)
        
        for t in range(tau):
            s = t / (tau - 1)
            pos = start + s * dist_vec
            xref = xref.at[0, t].set(pos[0])
            xref = xref.at[1, t].set(pos[1])
            xref = xref.at[2, t].set(pos[2])
            xref = xref.at[3, t].set(theta_ref)
            xref = xref.at[4, t].set(phi_ref)
            xref = xref.at[5, t].set(v_ref)
            
        p = Player(xref=xref, f=drone_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
        players.append(p)
        
    # Joint Constraints
    # r_col passed as argument
    u_bound = np.array([0.2, 1.0, 1.0]) # Adjusted control bounds
    
    def g_constraints(x_joint, u_joint):
        constraints = []
        
        # 1. Collision Avoidance (All pairs)
        for i in range(n_players):
            xi = x_joint[i*d:(i+1)*d]
            for j in range(i+1, n_players):
                xj = x_joint[j*d:(j+1)*d]
                # 3D distance
                dist = np.sqrt((xi[0]-xj[0])**2 + (xi[1]-xj[1])**2 + (xi[2]-xj[2])**2)
                constraints.append(-dist + r_col)
                
        # 2. Positive Velocity
        for i in range(n_players):
            xi = x_joint[i*d:(i+1)*d]
            constraints.append(-xi[5])
            
        # 3. Control Bounds
        for i in range(n_players):
            ui = u_joint[i*m:(i+1)*m]
            constraints.extend(ui - u_bound)
            constraints.extend(-ui - u_bound)
            
        if not constraints: return np.array([0.0])
        return np.array(constraints)
        
    game = PotientialGame(players=players, g=g_constraints)
    game.type = 'drone'
    game.configs = configs # Save for plotting
    return game

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(x_sol, game, r_col, dt):
    n = game.n
    d = 8
    tau = game.tau
    
    metrics = {}
    
    # 1. Vertical Airspace Used (Delta Z)
    # Extract all z coordinates (index 2)
    all_z = []
    for i in range(n):
        z_traj = x_sol[i*d + 2, :]
        all_z.extend(z_traj)
    
    all_z_arr = np.array(all_z)
    z_min = np.min(all_z_arr)
    z_max = np.max(all_z_arr)
    metrics['vertical_airspace_usage'] = float(z_max - z_min)
    
    # 2. Total Delay
    # Defined as Sum(Actual Arrival - Ideal Arrival).
    # Here horizon is fixed. So delay effectively means "distance from goal at final time" converted to time?
    # Or integrated velocity difference?
    # Let's measure "Time to Reach Goal Radius".
    goal_radius = 0.5
    total_delay = 0.0
    ideal_time = (tau - 1) * dt # In this setup, ideal is exactly tau
    
    # But if they slow down, they might not reach.
    # Let's use "Cost" as a proxy for delay/effort?
    # Prompt asks for "Total Delay (s)".
    # Let's calculate how much later they arrive compared to straight line flight.
    # Since tau is fixed, if they arrive at tau, delay is 0.
    # If they don't reach, delay is infinite?
    # Better metric: "Extra Time Required".
    # v_avg = dist / time. 
    # delay = (dist / v_avg_actual) - (dist / v_ref)
    
    for i in range(n):
        # Distance covered
        traj = x_sol[i*d : i*d+3, :] # x,y,z
        dist_covered = 0.0
        for t in range(tau-1):
            dist_covered += np.linalg.norm(traj[:, t+1] - traj[:, t])
            
        # Average speed
        avg_speed = dist_covered / ((tau-1)*dt)
        
        # Start-Goal dist
        start = np.array(game.configs[i]['start'])
        goal = np.array(game.configs[i]['goal'])
        sg_dist = np.linalg.norm(goal - start)
        
        ideal_t = sg_dist / 5.0 # V_ref ~ 5.0
        actual_t = sg_dist / avg_speed if avg_speed > 0.1 else tau*dt
        
        delay = max(0.0, actual_t - ideal_t)
        total_delay += delay
        
    metrics['total_delay'] = float(total_delay)
    
    # 3. Trajectory Entropy (Tortuosity)
    # Arc Length / Straight Line Distance
    total_tortuosity = 0.0
    for i in range(n):
        # Arc Length
        traj = x_sol[i*d : i*d+3, :]
        arc_len = 0.0
        for t in range(tau-1):
            arc_len += np.linalg.norm(traj[:, t+1] - traj[:, t])
            
        # Straight Line
        start = np.array(game.configs[i]['start'])
        goal = np.array(game.configs[i]['goal'])
        euc_dist = np.linalg.norm(goal - start)
        
        tortuosity = arc_len / euc_dist if euc_dist > 0 else 1.0
        total_tortuosity += tortuosity
        
    metrics['avg_tortuosity'] = float(total_tortuosity / n)
    
    return metrics

# ==========================================
# PLOTTING
# ==========================================

def plot_map_setup(game, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, cfg in enumerate(game.configs):
        start = cfg['start']
        goal = cfg['goal']
        color = cfg['color']
        
        # Plot Start
        ax.scatter(start[0], start[1], start[2], color=color, marker='o', s=100, label=f'Start {i+1}')
        ax.text(start[0], start[1], start[2], f'S{i+1}', color=color)
        
        # Plot Goal
        ax.scatter(goal[0], goal[1], goal[2], color=color, marker='x', s=100, label=f'Goal {i+1}')
        ax.text(goal[0], goal[1], goal[2], f'G{i+1}', color=color)
        
        # Draw ideal path line
        ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], '--', color=color, alpha=0.3)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Experiment 3 Setup: Converging 4')
    ax.legend()
    plt.savefig(os.path.join(output_dir, "experiment3_map.png"))
    plt.close()

def save_results(r_col, game, x_sol, output_dir):
    n = game.n
    d = 8
    tau = game.tau
    
    # 1. Static Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n):
        config = game.configs[i]
        color = config['color']
        traj = x_sol[i*d : i*d+3, :] # x,y,z
        
        ax.plot(traj[0, :], traj[1, :], traj[2, :], '.-', color=color, label=f"Drone {i+1}")
        ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0], color=color, marker='o') # Start
        ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1], color=color, marker='x') # End
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Altitude)')
    ax.set_title(f"Converging Drones: r_col = {r_col}")
    ax.legend()
    ax.set_zlim(0, 10)
    plt.savefig(os.path.join(output_dir, f"experiment3_traj_{r_col}.png"))
    plt.close()
    
    # 2. Animation (GIF)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    lines = [ax.plot([], [], [], 'o-', color=game.configs[i]['color'])[0] for i in range(n)]
    
    def update(frame):
        for i in range(n):
            traj = x_sol[i*d : i*d+3, :frame+1] # History up to frame
            lines[i].set_data(traj[0], traj[1])
            lines[i].set_3d_properties(traj[2])
        return lines
        
    ani = FuncAnimation(fig, update, frames=tau, blit=True)
    ani.save(os.path.join(output_dir, f"experiment3_{r_col}.gif"), writer='pillow', fps=10)
    plt.close()

# ==========================================
# MAIN
# ==========================================

def main():
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "experiment3")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    results = []
    
    print("Starting Experiment 3: The Volumetric Cost of Safety")
    
    # Create a sample game just to plot map
    sample_game = create_converging_drone_game(0.2, tau=40, dt=0.1)
    plot_map_setup(sample_game, artifacts_dir)
    
    for r_col in R_COL_VALUES:
        print(f"\nRunning r_col = {r_col}")
        
        game = create_converging_drone_game(r_col, tau=40, dt=0.1)
        
        # Solve
        seed = 42
        x_sol, u_sol, res = solve_game(game, seed=seed)
        
        if not res.success and "acceptable" not in str(res.message).lower():
            print(f"Warning: Solver did not converge for r_col = {r_col}")
            
        # Metrics
        met = calculate_metrics(x_sol, game, r_col, dt=0.1)
        met['r_col'] = r_col
        results.append(met)
        
        print(f"Metrics: {met}")
        
        # Save Plot
        save_results(r_col, game, x_sol, artifacts_dir)
        
        # Save Data (Trajectories & Controls)
        np.save(os.path.join(artifacts_dir, f"x_sol_{r_col}.npy"), onp.array(x_sol))
        np.save(os.path.join(artifacts_dir, f"u_sol_{r_col}.npy"), onp.array(u_sol))
        
    # Save CSV
    df = pd.DataFrame(results)
    cols = ['r_col', 'total_delay', 'vertical_airspace_usage', 'avg_tortuosity']
    df = df[cols]
    df.to_csv(os.path.join(artifacts_dir, "experiment3.csv"), index=False)
    print(f"\nExperiment 3 Completed. Results at {artifacts_dir}")

if __name__ == "__main__":
    main()

