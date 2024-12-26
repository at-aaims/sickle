# SICKLE 

SICKLE = Sparse Intelligent Curation for Knowledge-driven Learning Efficiency 
"Separating the wheat from the chaff"

# Subsampling

Login to Frontier

(no module purge - leave default loaded modules)

module load cray-python/3.10.10
source /lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate

OPENBLAS_NUM_THREADS=4 python subsample_maxent_seq.py --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64


# Training

module load cray-python/3.10.10 rocm

python -u train-pt-ddp.py --epochs 10 --patience 100 --dims 3 --dtype sst-binary --noseed -ns 10000 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 128 --window 5 --arch transformer

# See https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html
