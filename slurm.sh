#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
#SBATCH -p extended
##SBATCH -p batch  # partition
##SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.err

### Setup python environment

. environment

RUNDIR="$MEMBERWORK/stf218/sickle/${SLURM_JOB_ID}"
mkdir -p $RUNDIR "$RUNDIR/snapshots" "$RUNDIR/plots"

CASE=P1-Xmaxent-Yfull-Hmaxent-32.yaml

CASEPATH=config/SST/$CASE

cp $CASEPATH $RUNDIR
cp slurm.sh $RUNDIR  # for reproducibility
cd $RUNDIR

SRC=/lustre/orion/proj-shared/gen150/dsml/sickle

### SUBSAMPLING 

time srun -N $SLURM_NNODES -n56 python -u $SRC/subsample-mpi.py $CASE \
                           --output_dir $RUNDIR/snapshots >& $RUNDIR/subsample.out

### TRAINING

time srun -N $SLURM_NNODES --ntasks-per-node=8 python -u $SRC/train.py --plot \
                           --output_dir $RUNDIR/snapshots $CASE >& $RUNDIR/train.out
