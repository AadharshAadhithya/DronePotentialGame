import jax.numpy as np
import sys
import os

# Add parent directory to path to import classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from player import Player, PotientialGame

def create_car_game(tau=20, dt=0.1):
    # Dimensions
    d = 5 # x = [p, q, theta, v, omega]
    m = 2 # u = [delta_v, delta_omega]
    
    # Cost matrices
    Qhat = np.diag(np.array([50.0, 10.0, 5.0, 5.0, 2.0]))
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    Ri = np.diag(np.array([8.0, 4.0]))
    
    # Dynamics function
    def car_dynamics(x, u):
        # x = [p, q, theta, v, omega]
        # u = [delta_v, delta_omega]
        
        p, q, theta, v, omega = x
        delta_v, delta_omega = u
        
        p_next = p + dt * v * np.cos(theta)
        q_next = q + dt * v * np.sin(theta)
        theta_next = theta + dt * omega
        v_next = v + delta_v
        omega_next = omega + delta_omega
        
        return np.array([p_next, q_next, theta_next, v_next, omega_next])
    
    # Individual constraints (none specified, so dummy)
    def g_dummy(x, u):
        return np.array([0.0])
    
    # Reference trajectories
    # Player 1: Moving right on y=0
    xref1 = np.zeros((d, tau))
    # Initial state: p=0, q=0, theta=0, v=10, omega=0
    # Target: maintains v=10, moves along x axis
    t_range = np.arange(tau, dtype=float)
    xref1 = xref1.at[0, :].set(10 * t_range * dt)  # p
    xref1 = xref1.at[3, :].set(10.0)             # v

    # Player 2: Moving LEFT on y=0 (head-on collision course)
    # Start at x=20, move to x=0
    xref2 = np.zeros((d, tau))
    xref2 = xref2.at[0, :].set(20.0 - 10.0 * t_range * dt) # p decreases
    xref2 = xref2.at[2, :].set(np.pi)                      # theta = pi (facing left)
    xref2 = xref2.at[3, :].set(10.0)                       # v = 10 (speed)
        
    # Create players
    # Note: We need to pass Qi, Qtau, Ri to Player constructor
    # The Player class expects them to be passed in
    
    player1 = Player(xref=xref1, f=car_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
    player2 = Player(xref=xref2, f=car_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
    
    players = [player1, player2]
    
    # Joint constraint: Collision avoidance, Obstacle avoidance, Control bounds
    # r_col = 3
    r_col = 3.0
    
    # Obstacle
    x_obs = np.array([10.0, 0.0])
    r_obs = 4.0
    
    # Control bounds
    u_bound = np.array([0.15, 0.75]) # [delta_v, delta_omega]
    
    def g_constraints(x_joint, u_joint):
        # x_joint is (N*d,) = (10,)
        # x1 is x_joint[0:5], x2 is x_joint[5:10]
        x1 = x_joint[0:d]
        x2 = x_joint[d:2*d]
        
        # u_joint is (N*m,) = (4,)
        u1 = u_joint[0:m]
        u2 = u_joint[m:2*m]
        
        # 1. Collision avoidance between players
        p1, q1 = x1[0], x1[1]
        p2, q2 = x2[0], x2[1]
        dist_p1p2 = np.sqrt((p1 - p2)**2 + (q1 - q2)**2)
        c_collision = -dist_p1p2 + r_col
        
        # 2. Obstacle avoidance for P1
        dist_p1obs = np.sqrt((p1 - x_obs[0])**2 + (q1 - x_obs[1])**2)
        c_obs_p1 = -dist_p1obs + r_obs
        
        # 3. Obstacle avoidance for P2
        dist_p2obs = np.sqrt((p2 - x_obs[0])**2 + (q2 - x_obs[1])**2)
        c_obs_p2 = -dist_p2obs + r_obs
        
        # 4. Positive velocity constraint (-v <= 0)
        v1 = x1[3]
        v2 = x2[3]
        c_v1 = -v1
        c_v2 = -v2
        
        # 5. Control bounds (|u| - ub <= 0  =>  u - ub <= 0 AND -u - ub <= 0)
        # u1 bounds
        c_u1_upper = u1 - u_bound
        c_u1_lower = -u1 - u_bound
        
        # u2 bounds
        c_u2_upper = u2 - u_bound
        c_u2_lower = -u2 - u_bound
        
        # Concatenate all constraints
        # Shapes: scalar, scalar, scalar, scalar, scalar, (2,), (2,), (2,), (2,)
        return np.concatenate([
            np.array([c_collision, c_obs_p1, c_obs_p2, c_v1, c_v2]),
            c_u1_upper, c_u1_lower, c_u2_upper, c_u2_lower
        ])

    # Create game
    game = PotientialGame(players=players, g=g_constraints)
    
    # Store obstacle info in game object for plotting later
    game.obstacle = {'pos': x_obs, 'radius': r_obs}
    game.type = 'car'
    
    return game

def create_general_car_game(players_config, obstacles, tau=20, dt=0.1):
    """
    Creates a car game with variable number of players and obstacles.
    
    Args:
        players_config: List of dicts, each with 'start': (x,y), 'goal': (x,y)
        obstacles: List of dicts, each with 'x': float, 'y': float, 'r': float
        tau: Time horizon
        dt: Time step
    """
    # Dimensions
    d = 5 # x = [p, q, theta, v, omega]
    m = 2 # u = [delta_v, delta_omega]
    
    n_players = len(players_config)
    
    # Cost matrices
    Qhat = np.diag(np.array([50.0, 10.0, 5.0, 5.0, 2.0]))
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    Ri = np.diag(np.array([8.0, 4.0]))
    
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
        # Total time = (tau-1) * dt? 
        # Steps 0 to tau-1. Time 0 to (tau-1)*dt.
        # distance covered in (tau-1)*dt time.
        duration = (tau - 1) * dt
        ref_v = dist / duration if duration > 0 else 0.0
        
        # Steps 0 to 1
        s = t_range / (tau - 1)
        
        xref = xref.at[0, :].set(start[0] + s * dx)
        xref = xref.at[1, :].set(start[1] + s * dy)
        xref = xref.at[2, :].set(target_theta)
        xref = xref.at[3, :].set(ref_v) # Target speed matched to distance/time
        xref = xref.at[4, :].set(0.0)
        
        p = Player(xref=xref, f=car_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
        players.append(p)
        
    # Joint constraints
    r_col = 1.0
    u_bound = np.array([0.15, 0.75])
    
    def g_constraints(x_joint, u_joint):
        # x_joint: (N*d,)
        # u_joint: (N*m,)
        
        constraints = []
        
        # 1. Collision avoidance between players
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
            # ui - ub <= 0
            constraints.extend(ui - u_bound)
            # -ui - ub <= 0
            constraints.extend(-ui - u_bound)
            
        if not constraints:
            return np.array([0.0]) # Should not happen usually
            
        return np.array(constraints) # JAX will convert list to array
        
    game = PotientialGame(players=players, g=g_constraints)
    game.type = 'car'
    game.obstacles_list = obstacles # Store for plotting
    
    return game

if __name__ == "__main__":
    game = create_car_game()
    print("Car game created successfully!")
    print(f"Number of players: {game.n}")
    print(f"State dimension: {game.Qtau.shape[0]}")
    print(f"Control dimension: {game.Ri.shape[0]}")

