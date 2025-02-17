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

def take_snapshot(phase):
    jobid = os.environ.get('SLURM_JOBID', 'local_job')
    nodeid = os.environ.get('SLURM_NODEID', '0')
    hostname = os.uname().nodename
    output_dir = create_output_dir(jobid)
    snapshot_file = output_dir / f"{nodeid}_snapshot_{phase}.json"
    
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
    
    snapshot = {
        "timestamp": time.perf_counter(),
        "timestamp_readable": time.ctime(),  # human-readable
        "hostname": hostname,
        "phase": phase
    }
    
    for key, path in energy_files.items():
        try:
            with open(path) as f:
                raw_value = f.read().strip()
                # Debug print if needed:
                # print(f"DEBUG: Read from {path}: '{raw_value}'")
                # Extract the first token (the numeric energy value)
                numeric_str = raw_value.split()[0]
                snapshot[key] = float(numeric_str)
        except Exception as e:
            print(f"DEBUG: Error reading {path}: {e}")
            snapshot[key] = None

    # Save the snapshot to a JSON file
    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"Snapshot ({phase}) saved to {snapshot_file}")

def report_delta():
    jobid = os.environ.get('SLURM_JOBID', 'local_job')
    nodeid = os.environ.get('SLURM_NODEID', '0')
    output_dir = create_output_dir(jobid)
    
    start_file = output_dir / f"{nodeid}_snapshot_start.json"
    end_file = output_dir / f"{nodeid}_snapshot_end.json"
    
    try:
        with open(start_file) as f:
            start_snapshot = json.load(f)
        with open(end_file) as f:
            end_snapshot = json.load(f)
    except Exception as e:
        print("Error loading snapshots:", e)
        return
    
    delta = {}
    total_energy = 0.0
    # Energy keys for which we want to compute a total energy consumption
    energy_keys = ["CPU", "Mem", "Accel0", "Accel1", "Accel2", "Accel3"]
    
    for key in energy_keys:
        try:
            delta_value = end_snapshot[key] - start_snapshot[key]
            delta[key] = delta_value  # in Joules
            total_energy += delta_value
        except Exception as e:
            print(f"Error computing delta for {key}: {e}")
            delta[key] = None

    # Compute delta for Node energy
    try:
        node_delta = end_snapshot["Node"] - start_snapshot["Node"]
    except Exception as e:
        print("Error computing delta for Node energy:", e)
        node_delta = None

    # Compute elapsed time (in seconds) using the stored epoch timestamps
    try:
        elapsed_time = end_snapshot["timestamp"] - start_snapshot["timestamp"]
    except Exception as e:
        print("Error computing elapsed time:", e)
        elapsed_time = None

    # Compute average power if elapsed_time is valid and > 0
    if elapsed_time and elapsed_time > 0:
        avg_power = total_energy / elapsed_time  # in Watts (Joules per second)
    else:
        avg_power = None

    # Report
    print("Energy consumption during the run:")
    for key, value in delta.items():
        print(f"  {key}: {value} Joules")
    print(f"Total (CPU + Mem + Accel0 + Accel1 + Accel2 + Accel3): {total_energy} Joules")
    print(f"Node Energy Delta: {node_delta} Joules")
    if elapsed_time is not None:
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    else:
        print("Elapsed Time: not available")
    if avg_power is not None:
        print(f"Average Power: {avg_power:.2f} Watts")
    else:
        print("Average Power: not available")

def main():
    parser = argparse.ArgumentParser(description="Energy monitoring tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand for taking a snapshot
    parser_snapshot = subparsers.add_parser("snapshot", help="Take an energy snapshot")
    parser_snapshot.add_argument("phase", choices=["start", "end"],
                                 help="Phase of the measurement (start/end)")
    
    # Subcommand for reporting the delta
    parser_report = subparsers.add_parser("report", help="Report energy consumption and average power")
    
    args = parser.parse_args()
    
    if args.command == "snapshot":
        take_snapshot(args.phase)
    elif args.command == "report":
        report_delta()

if __name__ == "__main__":
    main()
