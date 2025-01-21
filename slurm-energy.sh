#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Load modules
module load cray-python/3.10.10
module load rocm/5.7.1
module load craype-accel-amd-gfx90a

#. ./venv-cray-python3.10.10/bin/activate  # Activate your own!

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

srun -N 2 -n 2 --ntasks-per-node=1 --overlap ./energy/power_dump.sh &
srun -N 2 -n 2 --ntasks-per-node=1 --overlap ./energy/read_energy.sh

srun --kill-on-bad-exit=1 --export=ALL python -u train-pt-ddp-multinode.py \
     --epochs 10 --patience 100 --dims 3 --dtype sst-binary \
     --noseed -ns 10000 \
     --input_vars u v w r --output_vars p --cluster_var pv \
     --nx 514 --ny 512 --nz 256 --gravity z \
     --nxsl 128 --nysl 128 --nzsl 128 \
     --window 5 --arch transformer

srun -N 2 -n 2 --ntasks-per-node=1 --overlap ./energy/read_energy.sh
scancel $SLURM_JOB_ID
