#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
##SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

CASE=config/P1F4R32/sample-reconstruction.yaml

### SUBSAMPLING 
module purge
source /lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate

module load PrgEnv-cray-amd
srun -N $SLURM_NODES -n 8 python subsample-mpi.py $CASE

### START ENERGY BENCHMARKING

srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap ./energy/power_dump.sh &
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap ./energy/read_energy.sh

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

srun -N $SLURM_NNODES -n8 python -u train-ddp-multinode.py $CASE

# Check power again
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap ./energy/read_energy.sh

#scancel $SLURM_JOB_ID
