#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
#SBATCH -p extended  # partition
##SBATCH -q debug # priority
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

RUNDIR=$MEMBERWORK/stf218/sickle-${SLURM_JOB_ID}
mkdir -p $RUNDIR $RUNDIR/snapshots $RUNDIR/plots
CASE=baseline-reconstruction.yaml
CASEPATH=config/exp0/$CASE

cp $CASEPATH $RUNDIR && cd $RUNDIR

SRC=/lustre/orion/proj-shared/gen150/dsml/sickle

### SUBSAMPLING 
module purge
source /lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate

# Take energy snapshot
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python $SRC/energy.py snapshot start

module load PrgEnv-cray-amd
time srun -N $SLURM_NNODES -n 8 python $SRC/subsample-mpi.py $CASE

### START ENERGY BENCHMARKING

# Take energy snapshot
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python $SRC/energy.py snapshot lap

### TRAINING

# Variables for DDP
WORLD_SIZE=$((SLURM_NTASKS))
NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(hostname -i)
export NCCL_SOCKET_IFNAME=hsn0
export MASTER_PORT=3442
export PYTORCH_ROCM_ARCH=gfx90a
# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

echo "World Size: $WORLD_SIZE, Node Rank: $NODE_RANK, Master Addr: $MASTER_ADDR, Master Port: $MASTER_PORT"

source '/lustre/orion/proj-shared/gen150/dsml/venv/pyt/bin/activate'
module load rocm/5.7.1

time srun -N $SLURM_NNODES -n 8 python -u $SRC/train-ddp-multinode.py $CASE

# Take energy snapshot
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python $SRC/energy.py snapshot end

# Generate energy usage report
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python $SRC/energy.py report
