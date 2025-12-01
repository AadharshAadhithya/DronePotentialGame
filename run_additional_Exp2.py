
import subprocess
import os
import sys
import pandas as pd
import glob

def run_additional_exp2():
    print("==================================================")
    print("       RUNNING ADDITIONAL EXP2 SCENARIOS          ")
    print("       (Cramped: 0.6 and 0.4 gaps)                ")
    print("==================================================")
    
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    exp2_script = "experiments/experiment2_4drone.py"
    
    # New scenarios added to experiment2_4drone.py
    scenarios = [
        "Very_Cramped_0.6",
        "Very_Cramped_0.4",
        "Blocked_Ambulance_0.6",
        "Blocked_Ambulance_0.4"
    ]

    procs = []
    files_to_close = []

    for sc in scenarios:
        print(f"Launching: {sc}")
        
        log_file_path = os.path.join(logs_dir, f"exp2_extra_{sc}.log")
        f = open(log_file_path, "w")
        files_to_close.append(f)
        
        cmd = [sys.executable, exp2_script, sc]
        p = subprocess.Popen(cmd, stdout=f, stderr=f)
        procs.append((p, sc))

    print(f"\nLaunched {len(procs)} processes. Waiting...")
    
    failed = []
    for p, sc in procs:
        p.wait()
        if p.returncode != 0:
            print(f"❌ {sc} FAILED. See logs/exp2_extra_{sc}.log")
            failed.append(sc)
        else:
            print(f"✅ {sc} COMPLETED")
            
    for f in files_to_close:
        f.close()

    if failed:
        print(f"\nFailures: {failed}")
    else:
        print("\nAll additional experiments completed.")

    # Aggregate (append to existing or create new?)
    # Let's just aggregate specifically these runs or re-aggregate everything?
    # The script `experiments/experiment2_4drone.py` saves partials to `artifacts/experiment2_4drone/metrics_*.csv`.
    # We can re-run the aggregation logic to include everything in that folder.
    print("\nAggregating All Exp2 Results...")
    aggregate_exp2()
    print("Done.")

def aggregate_exp2():
    artifacts_dir = os.path.join("artifacts", "experiment2_4drone")
    # This pattern picks up ALL metrics files, old and new, if they exist
    pattern = os.path.join(artifacts_dir, "metrics_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No metrics files found.")
        return

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        
        # If the main CSV already exists, we might want to merge with it?
        # But simpler strategy: The individual metrics_*.csv files represent the 'source of truth'.
        # If previous run cleaned them up, we can't re-aggregate them.
        # So we should probably check if `experiment2_4drone.csv` exists and append to it?
        
        main_csv = os.path.join(artifacts_dir, "experiment2_4drone.csv")
        if os.path.exists(main_csv):
            try:
                existing_df = pd.read_csv(main_csv)
                print(f"Merging with existing {main_csv} ({len(existing_df)} rows)...")
                full_df = pd.concat([existing_df, full_df], ignore_index=True)
                # Remove duplicates if any (based on condition)
                full_df = full_df.drop_duplicates(subset=['condition'], keep='last')
            except Exception as e:
                print(f"Could not read existing CSV: {e}")

        cols = ['condition', 'avg_velocity', 'min_dist_bottleneck', 'ambulance_velocity', 'traffic_velocity']
        cols = [c for c in cols if c in full_df.columns]
        full_df = full_df[cols]
        
        full_df.to_csv(main_csv, index=False)
        print(f"Saved updated Exp2 results to {main_csv}")
        
        # Clean up the NEW partial files
        for f in files:
            os.remove(f)

if __name__ == "__main__":
    run_additional_exp2()

