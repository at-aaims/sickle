#!/bin/bash
#SBATCH -A STF218
#SBATCH -J sickle
##SBATCH -p batch  # partition
#SBATCH -p extended  # partition
##SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/whbrewer/stf218/sickle/%j/%x_%j.err

### Setup python environment

source '/lustre/orion/proj-shared/gen150/dsml/venv/pyt/bin/activate'

RUNDIR="$MEMBERWORK/stf218/sickle/${SLURM_JOB_ID}"
mkdir -p $RUNDIR "$RUNDIR/snapshots" "$RUNDIR/plots"
#CASE=P1-Xsample-Yfull.yaml
CASE=P1-Xsample-Yfull-Hrandom.yaml
#CASE=P1-Xsample-Yfull-Hmaxent.yaml
#CASE=P1-Xfull-Yfull-Huniform.yaml -- doesn't work
#CASE=P1-Xfull-Yfull-Hrandom.yaml
#CASE=P1-Xfull-Yfull-Hmaxent.yaml

CASEPATH=config/SST/$CASE

cp $CASEPATH $RUNDIR
cp slurm.sh $RUNDIR  # for reproducibility
cd $RUNDIR

SRC=/lustre/orion/proj-shared/gen150/dsml/sickle

### SUBSAMPLING 

# Take energy snapshot
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python -u $SRC/energy.py snapshot start

time srun -N $SLURM_NNODES -n56 python -u $SRC/subsample-mpi.py $CASE \
                           --output_dir $RUNDIR/snapshots >& $RUNDIR/subsample.out

### START ENERGY BENCHMARKING

# Take energy snapshot
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python -u $SRC/energy.py snapshot lap

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

module load rocm/6.3.1 libfabric/1.22.0

time srun -N $SLURM_NNODES --ntasks-per-node=8 python -u $SRC/train.py --plot \
                                                --output_dir $RUNDIR/snapshots $CASE >& $RUNDIR/train.out

# Take energy snapshot
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python -u $SRC/energy.py snapshot end

# Generate energy usage report
srun -N $SLURM_NNODES --ntasks-per-node=1 --overlap python -u $SRC/energy.py report

# Aggregate report
python $SRC/energy.py aggregate
