import subprocess
import time

def get_rocm_smi_output():
    """Runs rocm-smi and returns the output as a string."""
    result = subprocess.run(['rocm-smi', '--showenergycounter', '--showpower', '--csv'], \
                            capture_output=True, text=True)
    return result.stdout

def parse_energy_and_power(rocm_output):
    """
    Parses the energy counter and accumulated energy (uJ) from the rocm-smi output.
    """
    lines = rocm_output.splitlines()
    accumulated_energy = []

    for line in lines:
        # Skip the header line
        if line.startswith("device") or line.startswith("card"):
            # Split CSV line and extract Accumulated Energy column
            columns = line.split(',')
            if len(columns) > 3 and columns[3].strip():  # Check if the Accumulated Energy column exists and is not empty
                try:
                    energy_value = float(columns[3].strip())
                    accumulated_energy.append(energy_value)
                except ValueError:
                    pass  # Handle potential parsing issues

    return accumulated_energy

def run_application(command):
    """Runs the machine learning application."""
    start_time = time.time()
    subprocess.run(command, shell=True)
    end_time = time.time()
    return end_time - start_time

def main():
    # Get initial energy readings
    initial_output = get_rocm_smi_output()
    initial_accumulated_energy = parse_energy_and_power(initial_output)

    # Run the machine learning application
    app_command = 'python -u train-pt-ddp.py --epochs 10 --patience 100 --dims 3 --dtype sst-binary --noseed -ns 10000 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 128 --window 5 --arch transformer'
    duration = run_application(app_command)

    # Get final energy readings
    final_output = get_rocm_smi_output()
    final_accumulated_energy = parse_energy_and_power(final_output)

    # Calculate energy consumption for each device
    energy_consumption = [
        final - initial
        for final, initial in zip(final_accumulated_energy, initial_accumulated_energy)
    ]

    # Display results for each device
    for idx, energy in enumerate(energy_consumption):
        energy_J = energy / 1e6  # Convert microjoules to joules
        print(f"Card {idx}: Energy Consumed: {energy_J:.2f} J")
        print(f"Card {idx}: Duration: {duration:.2f} seconds")
        print(f"Card {idx}: Average Power: {energy_J / duration:.2f} W")


if __name__ == "__main__":
    main()
