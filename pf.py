import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import squareform
from jax import vmap, jit
import jax.random as random
from functools import partial

def dbscan_from_dist_matrix(dist_matrix_condensed, num_points, eps, min_samples):
    """
    Custom implementation of DBSCAN using precomputed distance matrix.
    """
    # Convert condensed distance matrix to square form
    D = squareform(dist_matrix_condensed)
    
    labels = np.full(num_points, -1, dtype=int) # -1 represents noise
    cluster_id = 0
    visited = np.zeros(num_points, dtype=bool)
    
    for i in range(num_points):
        if visited[i]:
            continue
            
        visited[i] = True
        
        # Find neighbors
        neighbors = np.where(D[i] <= eps)[0]
        
        if len(neighbors) < min_samples:
            # Mark as noise (already -1)
            pass
        else:
            # Expand cluster
            labels[i] = cluster_id
            
            # Process queue
            queue = list(neighbors)
            # We can't modify list while iterating easily, so use index
            idx = 0
            while idx < len(queue):
                q = queue[idx]
                idx += 1
                
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = np.where(D[q] <= eps)[0]
                    if len(q_neighbors) >= min_samples:
                        # Add new neighbors to queue
                        # Avoid duplicates? set logic is cleaner but list + visited check works
                        for n in q_neighbors:
                            if n not in queue: # This check might be slow O(N)
                                queue.append(n)
                                
                if labels[q] == -1: # Noise or unassigned
                    labels[q] = cluster_id
                    
            cluster_id += 1
            
    return labels

def get_coarse_estimates(game, num_particles=100, noise_scale=0.1, alpha=10.0, cluster_threshold=2.0):
    """
    Implements Multi-Nash Particle Filter to find coarse estimates of local equilibria.
    
    Args:
        game: PotentialGame object
        num_particles: Number of particles
        noise_scale: Scale of process noise
        alpha: Barrier parameter
        cluster_threshold: Epsilon for DBSCAN
        
    Returns:
        List of initial guesses (x_traj, u_traj) for the solver.
    """
    
    # Dimensions
    # game.Qtau is (N*d, N*d). 
    n_players = game.n
    full_d = game.Qtau.shape[0] # Total state dimension
    full_m = game.Ri.shape[0]   # Total control dimension
    d = full_d // n_players
    m = full_m // n_players
    tau = game.tau
    
    # Augmented state dimension: x and u
    # \bar{x} = [x; u]
    n_aug = full_d + full_m
    
    # Measurement dimension
    # y = [x_ref; 0_constraints]
    # We need to determine the number of constraints.
    # We can run the constraint function once to check shapes.
    dummy_x = jnp.zeros(full_d)
    dummy_u = jnp.zeros(full_m)
    # Note: game.g expects single time step inputs usually? 
    # In car.py: g_constraints(x_joint, u_joint) returns 1D array.
    # But solve.py uses game.get_constraints which returns init, dyn, ineq.
    # game.g is the inequality constraints per time step.
    # We use game.g directly.
    dummy_g = game.g(dummy_x, dummy_u)
    n_constraints = dummy_g.shape[0]
    
    n_meas = full_d + n_constraints
    
    # Covariances
    # Process noise Q_bar: Random walk for both x and u?
    # Prompt says: x_{t+1} = f(x_t) + w_t. 
    # In virtual model: x_bar_{t+1} = f_bar(x_bar_t) + w_bar_t
    # We treat dynamics as "prediction" and noise as uncertainty.
    # For u, we use random walk: u_{t+1} = u_t + noise.
    # For x, we use dynamics: x_{t+1} = f(x_t, u_t) + noise.
    
    # Q_bar (Process Noise)
    Q_bar_diag = jnp.concatenate([
        jnp.ones(full_d) * (noise_scale**2), # Noise on state dynamics
        jnp.ones(full_m) * (noise_scale**2)  # Noise on control evolution
    ])
    Q_bar = jnp.diag(Q_bar_diag)
    
    # R_bar (Measurement Noise)
    # Measurement is [x_ref; 0].
    # x_ref part: standard deviation proportional to ...?
    # constraint part: standard deviation related to barrier?
    R_bar_diag = jnp.concatenate([
        jnp.ones(full_d) * 1.0,       # Measurement noise for reference tracking
        jnp.ones(n_constraints) * 0.1 # Measurement noise for constraints (tight)
    ])
    R_bar = jnp.diag(R_bar_diag)
    
    # Barrier function
    def barrier(z):
        # Smooth approximation of max(0, z)
        # z <= 0 is satisfied (barrier ~ 0)
        # z > 0 is violated (barrier > 0)
        return (1.0/alpha) * jnp.log(1.0 + jnp.exp(alpha * z))
        
    # Virtual Dynamics f_bar(x_bar)
    # x_bar = [x; u]
    # Returns next [x; u]
    def f_bar(x_bar):
        x = x_bar[:full_d]
        u = x_bar[full_d:]
        
        # Predict next state using game dynamics
        x_next = game.f(x, u)
        
        # Predict next control (Random Walk)
        u_next = u 
        
        return jnp.concatenate([x_next, u_next])
        
    # Virtual Measurement h_bar(x_bar)
    # Returns [x; barrier(g(x,u))]
    def h_bar(x_bar):
        x = x_bar[:full_d]
        u = x_bar[full_d:]
        
        g_val = game.g(x, u)
        g_bar = barrier(g_val)
        
        return jnp.concatenate([x, g_bar])
        
    # UKF Parameters
    kappa = 0.0 # Scaling parameter for UKF
    # Lambda = alpha^2 * (n + kappa) - n. Usually alpha=1e-3, beta=2
    ukf_alpha = 1e-3
    ukf_beta = 2.0
    ukf_lambda = ukf_alpha**2 * (n_aug + kappa) - n_aug
    
    # Weights for means and covariance
    w_m0 = ukf_lambda / (n_aug + ukf_lambda)
    w_c0 = w_m0 + (1 - ukf_alpha**2 + ukf_beta)
    w_i = 1.0 / (2 * (n_aug + ukf_lambda))
    
    weights_m = jnp.full(2 * n_aug + 1, w_i)
    weights_m = weights_m.at[0].set(w_m0)
    
    weights_c = jnp.full(2 * n_aug + 1, w_i)
    weights_c = weights_c.at[0].set(w_c0)
    
    # Initialize Particles
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    
    # Initial state distribution
    # Centered around x_ref[0] and u=0?
    x0_ref = game.xref[:, 0]
    u0_ref = jnp.zeros(full_m)
    mean_0 = jnp.concatenate([x0_ref, u0_ref])
    cov_0 = jnp.eye(n_aug) * (noise_scale * 5.0)**2 # Initial spread
    
    # Sample initial particles
    particles = random.multivariate_normal(subkey, mean_0, cov_0, shape=(num_particles,))
    
    # Store trajectories: (num_particles, tau, n_aug)
    trajectories = jnp.zeros((num_particles, tau, n_aug))
    trajectories = trajectories.at[:, 0, :].set(particles)
    
    # Loop over time
    current_particles = particles
    
    # Covariance per particle (Approximation: start with same covariance)
    # In full UKF-PF, we track covariance per particle.
    particle_covs = jnp.tile(cov_0[None, :, :], (num_particles, 1, 1))
    
    # Function to perform UKF Step for a single particle
    # We can VMAP this over particles
    def ukf_step_single(mean, cov, y_meas, key):
        # 1. Sigma Points Generation
        # Matrix square root of (n+lambda)*P
        # Use Cholesky
        factor = jnp.sqrt(n_aug + ukf_lambda)
        try:
            # Add small epsilon for stability
            L = jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(n_aug))
        except:
            L = jnp.eye(n_aug) * 1e-3 # Fallback
            
        sigmas = jnp.zeros((2 * n_aug + 1, n_aug))
        sigmas = sigmas.at[0].set(mean)
        for k in range(n_aug):
            sigmas = sigmas.at[k+1].set(mean + factor * L[:, k])
            sigmas = sigmas.at[n_aug+k+1].set(mean - factor * L[:, k])
            
        # 2. Prediction Step (Unscented Transform on Dynamics)
        sigmas_pred = vmap(f_bar)(sigmas)
        
        mean_pred = jnp.sum(weights_m[:, None] * sigmas_pred, axis=0)
        
        # Covariance pred
        diff = sigmas_pred - mean_pred[None, :]
        cov_pred = jnp.dot((weights_c[:, None] * diff).T, diff) + Q_bar
        
        # 3. Update Step
        # Re-calculate sigma points for predicted state (Standard UKF does this)
        # Or use the propagated ones if process noise is additive (Simplification)
        # Let's regenerate to be safe/standard
        try:
             L_pred = jnp.linalg.cholesky(cov_pred + 1e-6 * jnp.eye(n_aug))
        except:
             L_pred = jnp.eye(n_aug) * 1e-3
             
        sigmas_next = jnp.zeros((2 * n_aug + 1, n_aug))
        sigmas_next = sigmas_next.at[0].set(mean_pred)
        for k in range(n_aug):
            sigmas_next = sigmas_next.at[k+1].set(mean_pred + factor * L_pred[:, k])
            sigmas_next = sigmas_next.at[n_aug+k+1].set(mean_pred - factor * L_pred[:, k])
            
        # Transform to measurement space
        gammas = vmap(h_bar)(sigmas_next)
        y_pred_mean = jnp.sum(weights_m[:, None] * gammas, axis=0)
        
        diff_y = gammas - y_pred_mean[None, :]
        S = jnp.dot((weights_c[:, None] * diff_y).T, diff_y) + R_bar
        
        diff_x = sigmas_next - mean_pred[None, :]
        C_xz = jnp.dot((weights_c[:, None] * diff_x).T, diff_y)
        
        # Kalman Gain
        # K = C_xz @ inv(S)
        K = jnp.linalg.solve(S.T, C_xz.T).T
        
        # Updated Mean and Covariance
        innovation = y_meas - y_pred_mean
        mean_upd = mean_pred + jnp.dot(K, innovation)
        cov_upd = cov_pred - jnp.dot(K, jnp.dot(S, K.T))
        
        # 4. Implicit Sampling
        # Sample x ~ N(mean_upd, cov_upd)
        # gamma ~ N(0, I)
        gamma = random.normal(key, shape=(n_aug,))
        
        try:
            L_upd = jnp.linalg.cholesky(cov_upd + 1e-6 * jnp.eye(n_aug))
        except:
             L_upd = jnp.eye(n_aug) * 1e-3
        
        sample = mean_upd + jnp.dot(L_upd, gamma)
        
        return sample, cov_upd

    # Vmap over particles
    ukf_vmapped = vmap(ukf_step_single)
    
    # Time Loop
    print("Running PF...")
    for t in range(1, tau):
        # Define measurement for this step
        # y = [x_ref_t; 0]
        xref_t = game.xref[:, t]
        # 0 constraints
        y_meas = jnp.concatenate([xref_t, jnp.zeros(n_constraints)])
        
        # Split keys for particles
        key, subkey = random.split(key)
        subkeys = random.split(subkey, num_particles)
        
        # Run UKF step
        current_particles, particle_covs = ukf_vmapped(
            current_particles, 
            particle_covs, 
            jnp.tile(y_meas[None, :], (num_particles, 1)),
            subkeys
        )
        
        # Store
        trajectories = trajectories.at[:, t, :].set(current_particles)
        
    # Clustering
    print("Clustering...")
    # Convert to numpy for clustering
    trajs_np = np.array(trajectories) # (J, tau, n_aug)
    
    # Compute distance matrix
    # Using discrete Frechet distance
    # Since we have many particles, computing pairwise Frechet can be slow O(J^2 * tau^2).
    # With J=100, tau=20 -> 10000 pairs * 400 ops = 4M ops. Fast.
    
    def discrete_frechet(t1, t2):
        """
        Computes the discrete Frechet distance between two trajectories.
        t1, t2: (tau, dim) arrays
        """
        ca = np.ones((tau, tau)) * -1.0
        
        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            d = np.linalg.norm(t1[i] - t2[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(c(i-1, 0), d)
            elif i == 0 and j > 0:
                ca[i, j] = max(c(0, j-1), d)
            elif i > 0 and j > 0:
                ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), d)
            else:
                ca[i, j] = np.inf
            return ca[i, j]
            
        return c(tau-1, tau-1)

    # Vectorized/Faster Frechet?
    # For this size, iterative DP is better than recursive to avoid depth limits and overhead.
    def fast_frechet(t1, t2):
        # t1, t2: (tau, dim)
        mat = np.zeros((tau, tau))
        
        # Precompute distances
        # This part can be vectorized?
        # dists = cdist(t1, t2) # scipy.spatial.distance.cdist
        # But we want to stick to simple numpy
        pass 
        
        # Just use the recursive one but iterative
        ca = np.full((tau, tau), np.inf)
        
        # dists matrix
        # (tau, 1, dim) - (1, tau, dim) -> (tau, tau, dim)
        dists = np.linalg.norm(t1[:, None, :] - t2[None, :, :], axis=2)
        
        ca[0, 0] = dists[0, 0]
        for i in range(1, tau):
            ca[i, 0] = max(ca[i-1, 0], dists[i, 0])
        for j in range(1, tau):
            ca[0, j] = max(ca[0, j-1], dists[0, j])
            
        for i in range(1, tau):
            for j in range(1, tau):
                ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]), dists[i, j])
                
        return ca[tau-1, tau-1]

    # Compute condensed distance matrix
    # squareform requires a condensed matrix (1D)
    dist_matrix = []
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            # Only use state x for clustering, ignore u?
            # Or use full state? Usually spatial separation matters.
            # Let's use x only.
            traj1 = trajs_np[i, :, :full_d]
            traj2 = trajs_np[j, :, :full_d]
            d = fast_frechet(traj1, traj2)
            dist_matrix.append(d)
            
    dist_matrix = np.array(dist_matrix)
    
    # DBSCAN clustering
    # Default min_samples = 2 for small clusters? 
    # If we have 100 particles, we might expect few modes.
    min_samples = 3
    labels = dbscan_from_dist_matrix(dist_matrix, num_particles, eps=cluster_threshold, min_samples=min_samples)
    
    # Extract cluster means
    unique_labels = np.unique(labels)
    cluster_means = []
    
    # -1 is noise, ignore?
    # Or if everything is noise, we should probably warn.
    valid_clusters = unique_labels[unique_labels != -1]
    
    print(f"Found {len(valid_clusters)} clusters (and {np.sum(labels == -1)} noise points).")
    
    for lab in valid_clusters:
        indices = np.where(labels == lab)[0]
        # Average trajectories in this cluster
        mean_traj = np.mean(trajs_np[indices], axis=0)
        
        # Split into x and u
        x_sol = mean_traj[:, :full_d].T # (full_d, tau)
        u_sol = mean_traj[:, full_d:].T # (full_m, tau)
        
        cluster_means.append((x_sol, u_sol))
        
    # If no clusters found, maybe just take the global mean?
    if len(cluster_means) == 0 and len(labels) > 0:
        print("No clusters found, using global mean.")
        mean_traj = np.mean(trajs_np, axis=0)
        x_sol = mean_traj[:, :full_d].T
        u_sol = mean_traj[:, full_d:].T
        cluster_means.append((x_sol, u_sol))
        
    return cluster_means, trajs_np, dist_matrix
