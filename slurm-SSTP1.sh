#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
#SBATCH -p extended
##SBATCH -p batch  # partition
##SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.out
#SBATCH -j oe

# Define the list of cases
CASES=("P1-Xmaxent-Hmaxent-32" \
       "P1-Xrandom-Hrandom-32" \
       "P1-Xrandom-Hfull-32" \
       "P1-Xuips-Hrandom-32" \
       "P1-Xmaxent-Hrandom-32" \
       "P1-Xrandom-Hmaxent-32")

# Define the run directory and source path
RUNDIR="$MEMBERWORK/stf218/sickle/${SLURM_JOB_ID}"
SRC="/lustre/orion/proj-shared/gen150/dsml/sickle"

# Copy all case files to the run directory
for CASE in "${CASES[@]}"; do
    echo "Copying case file: $CASE.yaml"
    cp "config/SST/$CASE.yaml" "$RUNDIR"
done

# Copy the slurm.sh script for reproducibility
echo "Copying slurm.sh to $RUNDIR"
cp slurm.sh "$RUNDIR"

# Change directory to the run directory once
cd "$RUNDIR" || exit

# Initialize the counter variable
count=0

# Loop over each case and execute the commands
for CASE in "${CASES[@]}"; do
    echo "Processing case: $CASE"
    
    ### SUBSAMPLING
    time srun -N "$SLURM_NNODES" -n56 python -u "$SRC/subsample-mpi.py" "$CASE.yaml" \
              --output_dir "$RUNDIR/snapshots" >& "$RUNDIR/subsample${count}.out"

    ### TRAINING
    time srun -N "$SLURM_NNODES" --ntasks-per-node=8 python -u "$SRC/train.py" --plot \
              --output_dir "$RUNDIR/snapshots" "$CASE.yaml" >& "$RUNDIR/train${count}.out"

    echo "Finished processing case: $CASE"

    # Increment the counter
    count=$((count + 1))
done
