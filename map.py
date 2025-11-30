import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from car import create_general_car_game
from solve import solve_game

class MapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Game Map Editor")
        
        # Constants
        self.ROWS = 10
        self.COLS = 20
        self.CELL_SIZE = 30
        self.PHYSICS_SCALE = 1.0 # 1 grid unit = 1 physics unit
        
        # State
        self.grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)] # 0: Empty, 1: Wall
        self.players = [] # List of dicts: {'start': (r, c), 'goal': (r, c), 'color': 'blue'}
        self.mode = 'wall' # 'wall', 'p1_start', 'p1_goal', 'p2_start', ...
        
        # UI Layout
        self.create_widgets()
        
    def create_widgets(self):
        # Left Panel: Controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")
        
        ttk.Label(control_frame, text="Controls").pack(pady=5)
        
        # Mode selection
        self.mode_var = tk.StringVar(value="wall")
        
        ttk.Radiobutton(control_frame, text="Place Walls", variable=self.mode_var, value="wall").pack(anchor="w")
        
        self.player_frames = []
        self.add_player_btn = ttk.Button(control_frame, text="Add Player", command=self.add_player)
        self.add_player_btn.pack(pady=10)
        
        self.players_container = ttk.Frame(control_frame)
        self.players_container.pack(fill="x", expand=True)
        
        # Add 2 default players
        self.add_player()
        self.add_player()
        
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Run Button
        ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation).pack(pady=20, fill="x")
        
        ttk.Button(control_frame, text="Clear Walls", command=self.clear_walls).pack(pady=5, fill="x")
        
        # Right Panel: Grid Canvas
        canvas_frame = ttk.Frame(self.root, padding="10")
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        
        self.canvas = tk.Canvas(canvas_frame, width=self.COLS*self.CELL_SIZE, height=self.ROWS*self.CELL_SIZE, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        
        self.draw_grid()
        
    def add_player(self):
        pid = len(self.players)
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        color = colors[pid % len(colors)]
        
        self.players.append({'start': None, 'goal': None, 'color': color})
        
        p_frame = ttk.LabelFrame(self.players_container, text=f"Player {pid+1} ({color})")
        p_frame.pack(fill="x", pady=2)
        
        ttk.Radiobutton(p_frame, text="Set Start", variable=self.mode_var, value=f"p{pid}_start").pack(anchor="w")
        ttk.Radiobutton(p_frame, text="Set Goal", variable=self.mode_var, value=f"p{pid}_goal").pack(anchor="w")
        
    def clear_walls(self):
        self.grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.draw_grid()
        
    def on_click(self, event):
        self.handle_grid_event(event)
        
    def on_drag(self, event):
        # Only for wall drawing
        if self.mode_var.get() == 'wall':
            self.handle_grid_event(event)
            
    def handle_grid_event(self, event):
        c = event.x // self.CELL_SIZE
        r = event.y // self.CELL_SIZE
        
        if 0 <= r < self.ROWS and 0 <= c < self.COLS:
            mode = self.mode_var.get()
            
            if mode == 'wall':
                # Toggle or set? Let's set to wall.
                # If we want toggle we need click logic, for drag we want paint.
                # Let's just set to 1 (Wall). Right click could be clear?
                # For simplicity, simple toggle on click, paint on drag?
                # Let's just paint wall.
                self.grid[r][c] = 1
            elif mode.startswith('p'):
                # Parse p{id}_{type}
                parts = mode.split('_')
                pid = int(parts[0][1:])
                ptype = parts[1] # 'start' or 'goal'
                
                # Clear previous pos if exists
                # Actually, just update
                self.players[pid][ptype] = (r, c)
                
            self.draw_grid()
            
    def draw_grid(self):
        self.canvas.delete("all")
        
        # Draw grid lines
        for r in range(self.ROWS + 1):
            self.canvas.create_line(0, r*self.CELL_SIZE, self.COLS*self.CELL_SIZE, r*self.CELL_SIZE, fill="#ddd")
        for c in range(self.COLS + 1):
            self.canvas.create_line(c*self.CELL_SIZE, 0, c*self.CELL_SIZE, self.ROWS*self.CELL_SIZE, fill="#ddd")
            
        # Draw Walls
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] == 1:
                    x1 = c * self.CELL_SIZE
                    y1 = r * self.CELL_SIZE
                    x2 = x1 + self.CELL_SIZE
                    y2 = y1 + self.CELL_SIZE
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray")
                    
        # Draw Players
        for i, p in enumerate(self.players):
            color = p['color']
            # Start
            if p['start']:
                r, c = p['start']
                x = c * self.CELL_SIZE + self.CELL_SIZE/2
                y = r * self.CELL_SIZE + self.CELL_SIZE/2
                self.canvas.create_oval(x-8, y-8, x+8, y+8, fill=color, outline="black", width=2)
                self.canvas.create_text(x, y, text="S", fill="white", font=("Arial", 8, "bold"))
                
            # Goal
            if p['goal']:
                r, c = p['goal']
                x = c * self.CELL_SIZE + self.CELL_SIZE/2
                y = r * self.CELL_SIZE + self.CELL_SIZE/2
                self.canvas.create_rectangle(x-8, y-8, x+8, y+8, fill=color, outline="black", width=2)
                self.canvas.create_text(x, y, text="G", fill="white", font=("Arial", 8, "bold"))

    def grid_to_physics(self, r, c):
        # Grid: (0,0) top-left. Physics: (0,0) bottom-left?
        # car.py usually assumes standard Cartesian.
        # If grid is 10 rows, row 0 is y=9.5, row 9 is y=0.5?
        # Let's assume Physics origin is bottom-left of grid.
        
        # x = col * scale + half_cell
        # y = (ROWS - 1 - row) * scale + half_cell
        
        scale = self.PHYSICS_SCALE
        x = c * scale + scale / 2.0
        y = (self.ROWS - 1 - r) * scale + scale / 2.0
        return x, y

    def run_simulation(self):
        # Validate inputs
        active_players = []
        for i, p in enumerate(self.players):
            if p['start'] and p['goal']:
                start_pos = self.grid_to_physics(*p['start'])
                goal_pos = self.grid_to_physics(*p['goal'])
                active_players.append({
                    'start': start_pos,
                    'goal': goal_pos,
                    'color': p['color']
                })
        
        if not active_players:
            messagebox.showerror("Error", "Define at least one player with Start and Goal.")
            return
            
        # Collect obstacles
        obstacles = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] == 1:
                    x, y = self.grid_to_physics(r, c)
                    # Radius: slightly larger than half diagonal to prevent slipping through corners?
                    # Or just 0.6
                    obstacles.append({'x': x, 'y': y, 'r': 0.7})
        
        print(f"Starting simulation with {len(active_players)} players and {len(obstacles)} obstacles...")
        
        try:
            # Reverting to original faster settings (tau=20, dt=0.2)
            # Note: This may allow "tunneling" (jumping over thin walls) but is much faster.
            game = create_general_car_game(active_players, obstacles, tau=20, dt=0.2)
            
            # Generate a random seed for this run
            run_seed = np.random.randint(0, 10000)
            print(f"Running with seed: {run_seed}")
            x_sol, u_sol, res = solve_game(game, seed=run_seed, noise_scale=0.01)
            
            if not res.success and "acceptable" not in str(res.message).lower():
                messagebox.showwarning("Warning", f"Solver failed: {res.message}")
                
            self.animate_results(game, x_sol, active_players, obstacles)
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def animate_results(self, game, x_sol, players_config, obstacles):
        # Open a new window for animation
        # Or just use plt.show() which blocks?
        # Let's use plt.show()
        
        n = game.n
        d = game.Qtau.shape[0] // n
        tau = game.tau
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(0, self.COLS * self.PHYSICS_SCALE)
        ax.set_ylim(0, self.ROWS * self.PHYSICS_SCALE)
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Draw obstacles
        for obs in obstacles:
            circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='black', alpha=0.5)
            ax.add_patch(circle)
            
        # Initialize player objects
        player_lines = []
        player_dots = []
        
        for i, cfg in enumerate(players_config):
            color = cfg['color']
            line, = ax.plot([], [], color=color, linestyle='-', alpha=0.5)
            dot, = ax.plot([], [], color=color, marker='o')
            player_lines.append(line)
            player_dots.append(dot)
            
            # Draw Goal
            ax.plot(cfg['goal'][0], cfg['goal'][1], marker='x', color=color, markersize=10, markeredgewidth=2)
            # Draw Start
            ax.plot(cfg['start'][0], cfg['start'][1], marker='o', color=color, fillstyle='none')
            
        # Extract trajectories
        trajectories = [] # List of (2, tau) arrays
        for i in range(n):
            traj = x_sol[i*d : i*d + 2, :] # p, q
            trajectories.append(traj)
            
        def update(frame):
            for i in range(n):
                # Current history up to frame
                traj = trajectories[i]
                
                # Line
                player_lines[i].set_data(traj[0, :frame+1], traj[1, :frame+1])
                
                # Dot
                player_dots[i].set_data([traj[0, frame]], [traj[1, frame]])
                
            return player_lines + player_dots
            
        # Revert animation speed to match dt=0.2 (200ms)
        ani = FuncAnimation(fig, update, frames=tau, interval=200, blit=True, repeat=True)
        plt.title("Trajectory Optimization Result")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = MapGUI(root)
    root.mainloop()

