
import subprocess
import os
import sys
import time
import pandas as pd
import glob

def run_parallel():
    print("==================================================")
    print("       RUNNING ALL EXPERIMENTS IN PARALLEL        ")
    print("==================================================")
    
    # Define logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Define experiments and scenarios
    # Experiment 1
    exp1_script = "experiments/experiment1_4drone.py"
    exp1_scenarios = ["Low", "Baseline", "High", "Asymmetric"]
    
    # Experiment 2
    exp2_script = "experiments/experiment2_4drone.py"
    exp2_scenarios = [
        "Nominal", 
        "High", 
        "Very_Cramped", 
        "Blocked_Ambulance_Nominal", 
        "Blocked_Ambulance_Cramped"
    ]

    procs = []
    files_to_close = []

    # Launch Experiment 1 Scenarios
    for sc in exp1_scenarios:
        print(f"Launching Exp1: {sc}")
        
        log_file_path = os.path.join(logs_dir, f"exp1_{sc}.log")
        f = open(log_file_path, "w")
        files_to_close.append(f)
        
        cmd = [sys.executable, exp1_script, sc]
        # Run in background, redirect stdout/stderr
        p = subprocess.Popen(cmd, stdout=f, stderr=f)
        procs.append((p, "Exp1", sc))

    # Launch Experiment 2 Scenarios
    for sc in exp2_scenarios:
        print(f"Launching Exp2: {sc}")
        
        log_file_path = os.path.join(logs_dir, f"exp2_{sc}.log")
        f = open(log_file_path, "w")
        files_to_close.append(f)
        
        cmd = [sys.executable, exp2_script, sc]
        p = subprocess.Popen(cmd, stdout=f, stderr=f)
        procs.append((p, "Exp2", sc))

    print(f"\nLaunched {len(procs)} processes. Logs are being saved to {logs_dir}/")
    print("Waiting for completion...")
    
    # Wait for all
    failed = []
    for p, exp, sc in procs:
        p.wait()
        if p.returncode != 0:
            print(f"❌ {exp} - {sc} FAILED with code {p.returncode}. See logs/{exp.lower()}_{sc}.log")
            failed.append(f"{exp}-{sc}")
        else:
            print(f"✅ {exp} - {sc} COMPLETED")
            
    # Close log files
    for f in files_to_close:
        f.close()

    if failed:
        print(f"\nSome experiments failed: {failed}")
    else:
        print("\nAll experiments completed successfully.")

    # Aggregate Results
    print("\nAggregating Results...")
    aggregate_exp1()
    aggregate_exp2()
    print("Done.")

def aggregate_exp1():
    artifacts_dir = os.path.join("artifacts", "experiment1_4drone")
    pattern = os.path.join(artifacts_dir, "metrics_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No metrics files found for Exp1.")
        return

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        # Reorder/Select columns if needed, matching original
        cols = ['condition', 'ri_multiplier', 'min_distance', 'jerk', 'avg_time_to_goal', 'max_deviation']
        # Filter only if columns exist
        cols = [c for c in cols if c in full_df.columns]
        full_df = full_df[cols]
        
        out_path = os.path.join(artifacts_dir, "experiment1_4drone.csv")
        full_df.to_csv(out_path, index=False)
        print(f"Saved aggregated Exp1 results to {out_path}")

        # Clean up partial files
        for f in files:
            os.remove(f)

def aggregate_exp2():
    artifacts_dir = os.path.join("artifacts", "experiment2_4drone")
    pattern = os.path.join(artifacts_dir, "metrics_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No metrics files found for Exp2.")
        return

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        cols = ['condition', 'avg_velocity', 'min_dist_bottleneck', 'ambulance_velocity', 'traffic_velocity']
        cols = [c for c in cols if c in full_df.columns]
        full_df = full_df[cols]
        
        out_path = os.path.join(artifacts_dir, "experiment2_4drone.csv")
        full_df.to_csv(out_path, index=False)
        print(f"Saved aggregated Exp2 results to {out_path}")
        
        # Clean up partial files
        for f in files:
            os.remove(f)

if __name__ == "__main__":
    run_parallel()
