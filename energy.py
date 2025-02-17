#!/usr/bin/env python3
import argparse
import os
import time
import json
from pathlib import Path

def create_output_dir(jobid):
    output_dir = Path(f"{jobid}_Power")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def take_snapshot(label):
    jobid = os.environ.get('SLURM_JOBID', 'local_job')
    nodeid = os.environ.get('SLURM_NODEID', '0')
    hostname = os.uname().nodename
    output_dir = create_output_dir(jobid)
    snapshot_file = output_dir / f"{nodeid}_snapshot_{label}.json"
    
    # Define the energy counter file paths
    energy_files = {
        'CPU': '/sys/cray/pm_counters/cpu_energy',
        'Mem': '/sys/cray/pm_counters/memory_energy',
        'Accel0': '/sys/cray/pm_counters/accel0_energy',
        'Accel1': '/sys/cray/pm_counters/accel1_energy',
        'Accel2': '/sys/cray/pm_counters/accel2_energy',
        'Accel3': '/sys/cray/pm_counters/accel3_energy',
        'Node': '/sys/cray/pm_counters/energy'
    }
    
    # Use time.perf_counter() for high-resolution elapsed time measurement
    snapshot = {
        "timestamp": time.perf_counter(),        # high-resolution timer (seconds)
        "timestamp_readable": time.ctime(),        # human-readable system time
        "hostname": hostname,
        "phase": label
    }
    
    for key, path in energy_files.items():
        try:
            with open(path) as f:
                raw_value = f.read().strip()
                # Extract the first token (the numeric energy value)
                numeric_str = raw_value.split()[0]
                snapshot[key] = float(numeric_str)
        except Exception as e:
            print(f"DEBUG: Error reading {path}: {e}")
            snapshot[key] = None

    # Save the snapshot to a JSON file
    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"Snapshot ('{label}') saved to {snapshot_file}")

def report_snapshots():
    """
    Scans for all snapshots for the current node,
    sorts them by timestamp, and computes differences.
    """
    jobid = os.environ.get('SLURM_JOBID', 'local_job')
    nodeid = os.environ.get('SLURM_NODEID', '0')
    output_dir = create_output_dir(jobid)
    
    # Get list of snapshot files for this node (e.g., "0_snapshot_*.json")
    snapshot_files = sorted(output_dir.glob(f"{nodeid}_snapshot_*.json"))
    if not snapshot_files:
        print("No snapshot files found.")
        return
    
    snapshots = []
    for file in snapshot_files:
        try:
            with open(file) as f:
                snap = json.load(f)
                snapshots.append(snap)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Sort snapshots by their high-resolution timestamp
    snapshots.sort(key=lambda s: s["timestamp"])
    
    # List available snapshots
    print("Available snapshots:")
    for snap in snapshots:
        print(f"  Phase: {snap.get('phase', 'unknown')}, Time: {snap['timestamp_readable']}")
    
    # Now compute differences between consecutive snapshots
    print("\nReport for consecutive snapshot intervals:")
    energy_keys = ["CPU", "Mem", "Accel0", "Accel1", "Accel2", "Accel3"]
    
    for i in range(len(snapshots) - 1):
        s1 = snapshots[i]
        s2 = snapshots[i+1]
        elapsed_time = s2["timestamp"] - s1["timestamp"]
        total_energy = 0.0
        print(f"\nInterval: '{s1.get('phase')}' ({s1['timestamp_readable']}) -> '{s2.get('phase')}' ({s2['timestamp_readable']})")
        for key in energy_keys:
            try:
                delta_value = s2.get(key, 0) - s1.get(key, 0)
                total_energy += delta_value
                print(f"  {key}: {delta_value} Joules")
            except Exception as e:
                print(f"  Error computing delta for {key}: {e}")
        avg_power = total_energy / elapsed_time if elapsed_time > 0 else None
        print(f"  Total Energy: {total_energy} Joules")
        print(f"  Elapsed Time: {elapsed_time:.2f} seconds")
        if avg_power is not None:
            print(f"  Average Power: {avg_power:.2f} Watts")
        else:
            print("  Average Power: not available")
    
    # Optionally, also compute overall difference (first to last)
    overall_elapsed = snapshots[-1]["timestamp"] - snapshots[0]["timestamp"]
    overall_energy = 0.0
    for key in energy_keys:
        overall_energy += snapshots[-1].get(key, 0) - snapshots[0].get(key, 0)
    overall_avg_power = overall_energy / overall_elapsed if overall_elapsed > 0 else None
    
    print("\nOverall Summary (first snapshot to last snapshot):")
    print(f"  Overall Energy: {overall_energy} Joules")
    print(f"  Overall Elapsed Time: {overall_elapsed:.2f} seconds")
    if overall_avg_power is not None:
        print(f"  Overall Average Power: {overall_avg_power:.2f} Watts")
    else:
        print("  Overall Average Power: not available")

def main():
    parser = argparse.ArgumentParser(description="Energy monitoring tool with lap functionality")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Allow any label for snapshot
    parser_snapshot = subparsers.add_parser("snapshot", help="Take an energy snapshot with a custom label (e.g., start, lap, end)")
    parser_snapshot.add_argument("label", help="Label for the snapshot (e.g. start, lap, end)")
    
    # Report command to analyze snapshots
    parser_report = subparsers.add_parser("report", help="Report energy consumption between snapshots")
    
    args = parser.parse_args()
    
    if args.command == "snapshot":
        take_snapshot(args.label)
    elif args.command == "report":
        report_snapshots()

if __name__ == "__main__":
    main()
