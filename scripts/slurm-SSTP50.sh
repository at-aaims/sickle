#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
#SBATCH -p extended
##SBATCH -p batch  # partition
##SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.out

# Setup environment
. environment

# Define the list of cases
CASES=("Hrandom-Xrandom")

#CASES=("P1-Hmaxent-Xmaxent-32" \
#       "P1-Hmaxent-Xrandom-32" \
#       "P1-Hmaxent-Xuips-32" \
#       "P1-Hrandom-Xfull" \
#       "P1-Hrandom-Xmaxent-32" \
#       "P1-Hrandom-Xrandom-32" \
#       "P1-Hrandom-Xuips-32")

# Define the run directory and source path
RUNDIR="$MEMBERWORK/stf218/sickle/${SLURM_JOB_ID}"
mkdir -p $RUNDIR "$RUNDIR/snapshots" "$RUNDIR/plots"

SRC="/lustre/orion/proj-shared/gen150/dsml/sickle"

# Copy all case files to the run directory
for CASE in "${CASES[@]}"; do
    echo "Copying case file: $CASE.yaml"
    cp "config/SST/P50/$CASE.yaml" "$RUNDIR"
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
    time srun -N "$SLURM_NNODES" -n 32 python -u "$SRC/subsample.py" "$CASE.yaml" \
              --output_dir "$RUNDIR/snapshots" >& "$RUNDIR/subsample${count}.out"

    ### TRAINING
    time srun -N "$SLURM_NNODES" --ntasks-per-node=8 python -u "$SRC/train.py" --plot \
              --output_dir "$RUNDIR/snapshots" "$CASE.yaml" >& "$RUNDIR/train${count}.out"

    echo "Finished processing case: $CASE"

    # Increment the counter
    count=$((count + 1))
done
