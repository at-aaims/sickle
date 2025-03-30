#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
#SBATCH -p batch
##SBATCH -p extended
##SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -o /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.out

# Setup environment
. environment

# Define the list of cases
CASES=("Hmaxent-Xmaxent-32"
       "Hmaxent-Xmaxent-64")

# Define the run directory and source path
RUNDIR="$MEMBERWORK/stf218/sickle/${SLURM_JOB_ID}"
mkdir -p $RUNDIR "$RUNDIR/snapshots" "$RUNDIR/plots"

SRC="/lustre/orion/proj-shared/gen150/dsml/sickle"

# Copy all case files to the run directory
for CASE in "${CASES[@]}"; do
    echo "Copying case file: $CASE.yaml"
    cp "config/SST/P1/$CASE.yaml" "$RUNDIR"
done

# Copy the slurm.sh script for reproducibility
echo "Copying slurm.sh to $RUNDIR"
cp "${BASH_SOURCE[0]}" "$RUNDIR"

# Change directory to the run directory once
cd "$RUNDIR" || exit

# Initialize the counter variable
count=0

# Loop over each case and execute the commands
for CASE in "${CASES[@]}"; do
    echo "Processing case: $CASE"
    
    ### SUBSAMPLING
    time srun -N "$SLURM_NNODES" --ntasks-per-node=32 python -u "$SRC/subsample.py" "$CASE.yaml" \
              --output_dir "$RUNDIR/snapshots" >& "$RUNDIR/subsample${count}.out"

    ### TRAINING
    time srun -N "$SLURM_NNODES" --ntasks-per-node=8 python -u "$SRC/train.py" --plot \
              --output_dir "$RUNDIR/snapshots" "$CASE.yaml" >& "$RUNDIR/train${count}.out"

    echo "Finished processing case: $CASE"

    # Increment the counter
    count=$((count + 1))
done
