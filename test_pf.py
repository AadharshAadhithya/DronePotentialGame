import jax.numpy as np
import numpy as onp
from car import create_car_game
from pf import get_coarse_estimates, dbscan_from_dist_matrix
from solve import solve_game
import matplotlib.pyplot as plt

def main():
    print("Creating car game...")
    game = create_car_game()
    
    print("Running Multi-Nash Particle Filter...")
    # Initial run to get trajectories and distance matrix
    estimates, trajs_np, dist_matrix = get_coarse_estimates(game, num_particles=100, noise_scale=0.2, cluster_threshold=2.0)
    
    # Plot raw particles
    print(f"Plotting {len(trajs_np)} raw particles...")
    fig_raw = plt.figure(figsize=(10, 8))
    ax_raw = fig_raw.add_subplot(111)
    
    d = game.Qtau.shape[0] // game.n
    n_players = game.n
    colors = ['b', 'r', 'g', 'm']
    
    for i in range(len(trajs_np)):
        traj = trajs_np[i]
        # traj is (tau, n_aug)
        x_traj = traj[:, :game.Qtau.shape[0]].T # (full_d, tau)
        
        for p in range(n_players):
            xi = x_traj[p*d:(p+1)*d, :]
            c = colors[p % len(colors)]
            ax_raw.plot(xi[0, :], xi[1, :], color=c, alpha=0.1) # Low alpha for density
            
    # Plot obstacle
    if hasattr(game, 'obstacle'):
        obs = game.obstacle
        circle = plt.Circle((obs['pos'][0], obs['pos'][1]), obs['radius'], 
                            color='k', fill=False, linestyle=':', label='Obstacle')
        ax_raw.add_patch(circle)
        
    ax_raw.set_title("Raw Particle Trajectories (Before Clustering)")
    ax_raw.axis('equal')
    
    output_raw = "pf_particles_raw.png"
    fig_raw.savefig(output_raw)
    print(f"Raw particles plot saved to {output_raw}")
    
    # Analyze Distances
    print("\nAnalyzing Distance Matrix...")
    print(f"Distance range: {np.min(dist_matrix):.4f} to {np.max(dist_matrix):.4f}")
    print(f"Mean distance: {np.mean(dist_matrix):.4f}")
    
    # Plot Histogram of distances
    plt.figure()
    plt.hist(dist_matrix, bins=50)
    plt.title("Histogram of Pairwise Frechet Distances")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.savefig("dist_hist.png")
    print("Distance histogram saved to dist_hist.png")
    
    # K-distance plot (Elbow Method for DBSCAN)
    # min_samples = 3
    k = 3
    from scipy.spatial.distance import squareform
    D = squareform(dist_matrix)
    # Sort distances for each point and take the k-th nearest neighbor
    # axis=1, sort along rows
    sorted_dists = np.sort(D, axis=1)
    k_dists = sorted_dists[:, k] # k-th nearest neighbor
    k_dists_sorted = np.sort(k_dists)
    
    plt.figure()
    plt.plot(k_dists_sorted)
    plt.title(f"k-distance Graph (k={k})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-NN Distance")
    plt.grid(True)
    plt.savefig("k_distance_plot.png")
    print("k-distance plot saved to k_distance_plot.png")
    
    # Sweep eps
    print("\nSweeping epsilon...")
    eps_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
    cluster_counts = []
    best_labels = None
    best_eps = 5.0
    # We want the eps that gives a "stable" number of clusters or matches expected count (6)
    # Simple heuristic: Maximize clusters (before they merge into 1)
    max_clusters = 0
    target_clusters = 6
    
    for eps_test in eps_values:
        labels = dbscan_from_dist_matrix(dist_matrix, 100, eps=eps_test, min_samples=3)
        unique_l = np.unique(labels)
        # Ignore noise -1
        n_clusters = len(unique_l[unique_l != -1])
        noise_points = np.sum(labels == -1)
        cluster_counts.append(n_clusters)
        print(f"eps={eps_test}: {n_clusters} clusters, {noise_points} noise")
        
        # Heuristic: Prioritize getting close to target if specified, otherwise max clusters
        if n_clusters >= max_clusters and n_clusters > 0:
            # If we have same number of clusters but fewer noise points, that's usually better?
            # Actually, if we merge distinct clusters, n_clusters drops.
            # If we increase eps and n_clusters stays same, it's stable.
            max_clusters = n_clusters
            best_labels = labels
            best_eps = eps_test
            
    print(f"\nSelected eps={best_eps} with {max_clusters} clusters.")
    
    # Plot clusters vs eps
    plt.figure()
    plt.plot(eps_values, cluster_counts, 'o-')
    plt.title("Number of Clusters vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Number of Clusters")
    plt.grid(True)
    plt.savefig("clusters_vs_eps.png")
    print("Cluster sweep plot saved to clusters_vs_eps.png")

    # Extract estimates using best_labels
    if max_clusters > 0:
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
        estimates = []

    print(f"Finalizing {len(estimates)} estimates.")
    
    if len(estimates) == 0:
        print("No estimates found!")
        return
    
    # Prepare plot for refined solutions
    num_plots = len(estimates)
    cols = min(num_plots, 3)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # Helper to plot
    def plot_traj(ax, x, u, title, is_guess=False):
        d = game.Qtau.shape[0] // game.n
        n_players = game.n
        
        colors = ['b', 'r', 'g', 'm']
        
        for i in range(n_players):
            xi = x[i*d:(i+1)*d, :]
            c = colors[i % len(colors)]
            style = '--' if is_guess else '.-'
            alpha = 0.4 if is_guess else 1.0
            label = f'P{i+1} Guess' if is_guess else f'P{i+1} Sol'
            
            ax.plot(xi[0, :], xi[1, :], style, color=c, alpha=alpha, label=label if not is_guess else None)
            
            if not is_guess:
                ax.plot(xi[0, 0], xi[1, 0], 'o', color=c)
                ax.plot(xi[0, -1], xi[1, -1], 'x', color=c)
            
        # Plot Obstacle
        if hasattr(game, 'obstacle'):
            obs = game.obstacle
            circle = plt.Circle((obs['pos'][0], obs['pos'][1]), obs['radius'], 
                                color='k', fill=False, linestyle=':', label='Obstacle')
            ax.add_patch(circle)
            
        ax.set_title(title)
        if not is_guess:
            ax.grid(True)
            ax.axis('equal')
            # ax.legend()

    # Solve for each estimate
    for i, (x_est, u_est) in enumerate(estimates):
        print(f"\nRefining estimate {i+1}...")
        
        # Plot guess first
        if i < len(axes):
            ax = axes[i]
            plot_traj(ax, x_est, u_est, "Guess", is_guess=True)
            
            # Solve
            try:
                x_sol, u_sol, res = solve_game(game, warm_start=(x_est, u_est))
                status = "Success" if res.success else "Failed"
                if hasattr(res, 'message') and "acceptable" in str(res.message).lower():
                    status = "Acceptable"
                    
                print(f"  {status}! Cost: {res.fun}")
                plot_traj(ax, x_sol, u_sol, f"Mode {i+1}\n{status}, Cost: {res.fun:.2f}")
                
            except Exception as e:
                print(f"  Error solving: {e}")
                ax.set_title(f"Mode {i+1} - Error")
            
    plt.tight_layout()
    output_file = "pf_multinash_results.png"
    plt.savefig(output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
