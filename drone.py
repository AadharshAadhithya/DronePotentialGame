import jax.numpy as np
import sys
import os

# Add parent directory to path to import classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from player import Player, PotientialGame

def create_drone_game(tau=30, dt=0.1):
    # Dimensions
    # State: [p, q, r, theta, phi, v, omega_theta, omega_phi]
    # p,q,r: position
    # theta: yaw (xy plane angle)
    # phi: pitch (angle from xy plane)
    # v: speed
    # omega_theta: yaw rate
    # omega_phi: pitch rate
    d = 8 
    
    # Control: [delta_v, delta_omega_theta, delta_omega_phi]
    m = 3 
    
    # Cost matrices (diagonal)
    # Weights: pos(3), ang(2), v(1), rates(2)
    diag_entries = np.array([50.0, 50.0, 50.0, # pos
                            10.0, 10.0,       # angles
                            5.0,              # v
                            5.0, 2.0])        # rates
    
    Qhat = np.diag(diag_entries)
    Qi = 0.6 * Qhat
    Qtau = 100 * Qhat
    
    # Control weights
    Ri = np.diag(np.array([8.0, 4.0, 4.0]))
    
    # Dynamics function
    def drone_dynamics(x, u):
        p, q, r, theta, phi, v, omega_theta, omega_phi = x
        delta_v, delta_omega_theta, delta_omega_phi = u
        
        # 3D unicycle / point mass kinematics
        p_next = p + dt * v * np.cos(theta) * np.cos(phi)
        q_next = q + dt * v * np.sin(theta) * np.cos(phi)
        r_next = r + dt * v * np.sin(phi)
        
        theta_next = theta + dt * omega_theta
        phi_next = phi + dt * omega_phi
        
        v_next = v + delta_v
        omega_theta_next = omega_theta + delta_omega_theta
        omega_phi_next = omega_phi + delta_omega_phi
        
        return np.array([p_next, q_next, r_next, theta_next, phi_next, v_next, omega_theta_next, omega_phi_next])
    
    # Dummy individual constraint
    def g_dummy(x, u):
        return np.array([0.0])
    
    # Reference trajectories
    # Player 1: (-0.5,-0.5,0) -> (0.5,0.5,0)
    # Distance approx 1.41. Time 2s.
    xref1 = np.zeros((d, tau))
    t_range = np.arange(tau, dtype=float)
    
    # Linear interpolation for position
    # Start -0.5, End 0.5
    # step = 1.0 / (tau-1) * t
    steps = t_range / (tau-1)
    xref1 = xref1.at[0, :].set(-0.5 + 1.0 * steps) # p
    xref1 = xref1.at[1, :].set(-0.5 + 1.0 * steps) # q
    xref1 = xref1.at[2, :].set(2.5 * steps)        # r: 0 -> 1
    
    # Angles for diagonal movement (1,1,0) direction
    # theta = 45 deg = pi/4
    # phi = 0
    xref1 = xref1.at[3, :].set(np.pi/4)       # theta
    xref1 = xref1.at[4, :].set(0.0)           # phi
    xref1 = xref1.at[5, :].set(1.0)           # v (reduced speed)
    
    # Player 2: (0.5,0.5,0) -> (-0.5,-0.5,1)
    xref2 = np.zeros((d, tau))
    xref2 = xref2.at[0, :].set(0.5 - 1.0 * steps) # p
    xref2 = xref2.at[1, :].set(0.5 - 1.0 * steps) # q
    xref2 = xref2.at[2, :].set(2.5 * steps)       # r: 0 -> 1
    
    # Angles for opposite diagonal (-1,-1,0)
    # theta = -135 deg = -3pi/4
    # phi = 0
    xref2 = xref2.at[3, :].set(-3*np.pi/4)    # theta
    xref2 = xref2.at[4, :].set(0.0)           # phi
    xref2 = xref2.at[5, :].set(1.0)           # v
    
    player1 = Player(xref=xref1, f=drone_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
    player2 = Player(xref=xref2, f=drone_dynamics, g=g_dummy, tau=tau, Qi=Qi, Qtau=Qtau, Ri=Ri)
    
    players = [player1, player2]
    
    # Joint constraints
    r_col = 0.2 # Scaled down
    
    # Cylindrical Obstacle
    # Infinite cylinder at (0, 0, z)
    obs_xy = np.array([0.0, 0.0])
    r_obs = 0.2 # Scaled down
    
    # Control bounds
    u_bound = np.array([0.15, 0.75, 0.75]) 
    
    def g_constraints(x_joint, u_joint):
        x1 = x_joint[0:d]
        x2 = x_joint[d:2*d]
        
        u1 = u_joint[0:m]
        u2 = u_joint[m:2*m]
        
        # 1. Collision avoidance (3D distance)
        p1, q1, r1 = x1[0], x1[1], x1[2]
        p2, q2, r2 = x2[0], x2[1], x2[2]
        dist_p1p2 = np.sqrt((p1 - p2)**2 + (q1 - q2)**2 + (r1 - r2)**2)
        c_collision = -dist_p1p2 + r_col
        
        # 2. Cylindrical Obstacle Avoidance (2D distance in xy plane)
        dist_p1obs = np.sqrt((p1 - obs_xy[0])**2 + (q1 - obs_xy[1])**2)
        c_obs_p1 = -dist_p1obs + r_obs
        
        dist_p2obs = np.sqrt((p2 - obs_xy[0])**2 + (q2 - obs_xy[1])**2)
        c_obs_p2 = -dist_p2obs + r_obs
        
        # 3. Positive velocity
        c_v1 = -x1[5]
        c_v2 = -x2[5]
        
        # 4. Control bounds
        c_u1_upper = u1 - u_bound
        c_u1_lower = -u1 - u_bound
        c_u2_upper = u2 - u_bound
        c_u2_lower = -u2 - u_bound
        
        return np.concatenate([
            np.array([c_collision, c_obs_p1, c_obs_p2, c_v1, c_v2]),
            c_u1_upper, c_u1_lower, c_u2_upper, c_u2_lower
        ])
        
    game = PotientialGame(players=players, g=g_constraints)
    game.obstacle = {'type': 'cylinder', 'pos': obs_xy, 'radius': r_obs}
    game.type = 'drone' # Mark as drone game
    
    return game

