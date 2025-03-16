#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import time
from pathlib import Path

def create_output_dir(jobid):
    output_dir = Path(f"{jobid}_Power")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def get_calling_filename():
    # Get the filename of the caller (one level up in the stack)
    caller_file = inspect.stack()[1].filename
    # Extract the basename without extension
    base_name = os.path.splitext(os.path.basename(caller_file))[0]
    return base_name

def create_output_dir(jobid, benchmark_name=None):
    """
    Creates an output directory based on the benchmark name.
    If benchmark_name is not provided, it uses the calling file's name.
    """
    if benchmark_name is None:
        benchmark_name = get_calling_filename()
    # You can include the jobid if needed
    output_dir = Path(f"{jobid}_{benchmark_name}_Power")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def take_snapshot(label, benchmark_name=None):
    jobid = os.environ.get('SLURM_JOBID', 'local_job')
    nodeid = os.environ.get('SLURM_NODEID', '0')
    hostname = os.uname().nodename
    output_dir = create_output_dir(jobid, benchmark_name)
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
        "timestamp_readable": time.ctime(),      # human-readable system time
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
    Reports detailed breakdown for a single node.
    (This function remains per-node.)
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
    
    print("\nDetailed Breakdown by Interval:")
    energy_keys = ["CPU", "Mem", "Accel0", "Accel1", "Accel2", "Accel3"]
    
    overall_total_energy = 0.0
    
    for i in range(len(snapshots) - 1):
        s1 = snapshots[i]
        s2 = snapshots[i+1]
        interval_time = s2["timestamp"] - s1["timestamp"]
        interval_total_energy = 0.0
        
        print(f"\nInterval: '{s1.get('phase')}' ({s1['timestamp_readable']}) -> '{s2.get('phase')}' ({s2['timestamp_readable']})")
        print(f"  Elapsed Time: {interval_time:.2f} seconds")
        
        for key in energy_keys:
            try:
                delta_value = s2.get(key, 0) - s1.get(key, 0)
                interval_total_energy += delta_value
                print(f"  {key}: {delta_value:.2f} Joules")
            except Exception as e:
                print(f"  Error computing delta for {key}: {e}")
        
        overall_total_energy += interval_total_energy
        if interval_time > 0:
            interval_avg_power = interval_total_energy / interval_time
        else:
            interval_avg_power = None
        
        print(f"  Total Energy for interval: {interval_total_energy:.2f} Joules")
        if interval_avg_power is not None:
            print(f"  Average Power for interval: {interval_avg_power:.2f} Watts")
        else:
            print("  Average Power for interval: not available")
    
    overall_elapsed = snapshots[-1]["timestamp"] - snapshots[0]["timestamp"]
    print("\nOverall Summary (first snapshot to last snapshot):")
    print(f"  Overall Energy Consumed: {overall_total_energy:.2f} Joules")
    print(f"  Overall Elapsed Time: {overall_elapsed:.2f} seconds")
    if overall_elapsed > 0:
        overall_avg_power = overall_total_energy / overall_elapsed
        print(f"  Overall Average Power: {overall_avg_power:.2f} Watts")
    else:
        print("  Overall Average Power: not available")

def aggregate_reports():
    """
    Aggregates snapshot data from all nodes in the output directory.
    It groups files by node id (the prefix of the file name),
    computes each node's overall energy consumption (first to last snapshot),
    and then computes an overall cluster summary.
    Additionally, it provides a breakdown between CPU energy (CPU + Mem)
    and GPU energy (Accel0 - Accel3).
    """
    jobid = os.environ.get('SLURM_JOBID', 'local_job')
    output_dir = create_output_dir(jobid)
    
    # Get all snapshot files (e.g., "0_snapshot_*.json", "1_snapshot_*.json", etc.)
    snapshot_files = sorted(output_dir.glob("*_snapshot_*.json"))
    if not snapshot_files:
        print("No snapshot files found for aggregation.")
        return
    
    # Group snapshots by node id (assumes filename starts with node id)
    node_snapshots = {}
    for file in snapshot_files:
        # file name format: "<nodeid>_snapshot_<label>.json"
        node_id = file.name.split('_')[0]
        if node_id not in node_snapshots:
            node_snapshots[node_id] = []
        try:
            with open(file) as f:
                snap = json.load(f)
                node_snapshots[node_id].append(snap)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    overall_cluster_energy = 0.0
    overall_cpu_energy = 0.0
    overall_gpu_energy = 0.0
    cluster_start = None
    cluster_end = None
    
    print("Per-Node Energy Consumption:")
    for node_id, snaps in node_snapshots.items():
        if not snaps:
            continue
        snaps.sort(key=lambda s: s["timestamp"])
        node_start = snaps[0]["timestamp"]
        node_end = snaps[-1]["timestamp"]
        if cluster_start is None or node_start < cluster_start:
            cluster_start = node_start
        if cluster_end is None or node_end > cluster_end:
            cluster_end = node_end
        
        # Calculate CPU energy: sum of CPU and Mem deltas
        cpu_energy = (
            (snaps[-1].get("CPU", 0) - snaps[0].get("CPU", 0)) +
            (snaps[-1].get("Mem", 0) - snaps[0].get("Mem", 0))
        )
        # Calculate GPU energy: sum of Accelerators' deltas
        gpu_energy = (
            (snaps[-1].get("Accel0", 0) - snaps[0].get("Accel0", 0)) +
            (snaps[-1].get("Accel1", 0) - snaps[0].get("Accel1", 0)) +
            (snaps[-1].get("Accel2", 0) - snaps[0].get("Accel2", 0)) +
            (snaps[-1].get("Accel3", 0) - snaps[0].get("Accel3", 0))
        )
        node_total_energy = cpu_energy + gpu_energy
        overall_cluster_energy += node_total_energy
        overall_cpu_energy += cpu_energy
        overall_gpu_energy += gpu_energy
        
        elapsed_node = node_end - node_start
        if elapsed_node > 0:
            avg_power_node = node_total_energy / elapsed_node
        else:
            avg_power_node = None
        
        print(f"\nNode {node_id}:")
        print(f"  Total Energy Consumed: {node_total_energy:.2f} Joules")
        print(f"    CPU Energy (CPU + Mem): {cpu_energy:.2f} Joules")
        print(f"    GPU Energy (Accel0 - Accel3): {gpu_energy:.2f} Joules")
        print(f"  Elapsed Time: {elapsed_node:.2f} seconds")
        if avg_power_node is not None:
            print(f"  Average Power: {avg_power_node:.2f} Watts")
        else:
            print("  Average Power: not available")
    
    # Compute overall cluster elapsed time (from earliest start to latest end)
    if cluster_start is not None and cluster_end is not None:
        cluster_elapsed = cluster_end - cluster_start
    else:
        cluster_elapsed = 0.0
    
    print("\nOverall Cluster Summary:")
    print(f"  Total Energy Consumed: {overall_cluster_energy:.2f} Joules")
    print(f"    Total CPU Energy (CPU + Mem): {overall_cpu_energy:.2f} Joules")
    print(f"    Total GPU Energy (Accel0 - Accel3): {overall_gpu_energy:.2f} Joules")
    print(f"  Cluster Elapsed Time: {cluster_elapsed:.2f} seconds")
    if cluster_elapsed > 0:
        overall_avg_power = overall_cluster_energy / cluster_elapsed
        print(f"  Overall Average Power: {overall_avg_power:.2f} Watts")
    else:
        print("  Overall Average Power: not available")


def main():
    parser = argparse.ArgumentParser(description="Energy monitoring tool with lap and aggregation functionality")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Allow any label for snapshot
    parser_snapshot = subparsers.add_parser("snapshot", help="Take an energy snapshot with a custom label (e.g., start, lap, end)")
    parser_snapshot.add_argument("label", help="Label for the snapshot (e.g. start, lap, end)")
    
    # Report command for per-node report
    parser_report = subparsers.add_parser("report", help="Report energy consumption between snapshots for this node")
    
    # Aggregate command for cross-node aggregation
    parser_aggregate = subparsers.add_parser("aggregate", help="Aggregate energy consumption reports across nodes")
    
    args = parser.parse_args()
    
    if args.command == "snapshot":
        take_snapshot(args.label)
    elif args.command == "report":
        report_snapshots()
    elif args.command == "aggregate":
        aggregate_reports()

if __name__ == "__main__":
    main()
