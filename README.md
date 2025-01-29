# SICKLE 

SICKLE = Sparse Intelligent Curation frameworK for Learning Efficiency 

SICKLE is a tool to "separate the wheat from the chaff", that is, to extract 
data with the probabilistically highest information content to improve the 
cost of training large models.

# Subsampling

Login to Frontier

    source /lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate

    python subsample.py --method maxent --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 10000 --input_vars u v w r --target p_full --output_vars p pv --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 32 --nysl 32 --nzsl 32 --window 2

# Training

    source '/lustre/orion/proj-shared/gen150/dsml/venv/pyt/bin/activate'

    module load cray-python/3.10.10 rocm

    python -u train-ddp.py --epochs 10 --patience 100 --dims 3 --dtype sst-binary --noseed -ns 10000 --input_vars u v w r --target p_full --output_vars p pv --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 32 --nysl 32 --nzsl 32 --window 2 --arch MLP_transformer --sequence

See https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html

# Tests on laptop - random and maxent

    python subsample.py -m random --path ~/data/cylinder --target drag -ns 540
    python subsample.py -m maxent --path ~/data/cylinder --target drag -ns 540 -cv p

# Tests on Frontier

    source '/lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate'

    python subsample.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

    python subsample.py -m full --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z

# Parallel tests on Frontier

    # OpenFOAM dataset - random and maxent

    srun -n 4 python -u subsample-mpi.py -m random --path ~/data/cylinder --target drag -ns 540
    srun -n 4 python -u subsample-mpi.py -m maxent --path ~/data/cylinder --target drag -ns 540

    # Taylor Green 10 timesteps
    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

    # Taylor Green 126 timesteps
    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data/P1F4R32_training/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

    # Green - isotropic homogenous 64T dataset

    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data/Green/ --plot -ns 1000 --input_vars u v w r --output_vars p --cluster_var r --nx 12290 --ny 6144 --nz 12288 --gravity y --nxsl 1536 --ny 768 --nzsl 1536 --nbytes 32 --timesteps 0.5 1.0

    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data/Green/ --plot -ns 1000 --input_vars u v w r --output_vars p --cluster_var r --nx 11202 --ny 5600 --nz 11200 --gravity y --nxsl 1536 --ny 768 --nzsl 1536 --nbytes 32 --timesteps 2.0

    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data/Green/ --plot -ns 1000 --input_vars u v w r --output_vars p --cluster_var r --nx 8450 --ny 4224 --nz 8448 --gravity y --nxsl 1536 --ny 768 --nzsl 1536 --nbytes 32 --timesteps 4.0

    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data/Green/ --plot -ns 1000 --input_vars u v w r --output_vars p --cluster_var r --nx 7682 --ny 3840 --nz 7680 --gravity y --nxsl 1536 --ny 768 --nzsl 1536 --nbytes 32 --timesteps 6.0

    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data/Green/ --plot -ns 1000 --input_vars u v w r --output_vars p --cluster_var r --nx 7170 --ny 3584 --nz 7168 --gravity y --nxsl 1536 --ny 768 --nzsl 1536 --nbytes 32 --timesteps 7.7

# Using with flow over cylinder case

    python subsample.py -m maxent --path ~/data/cylinder --target drag -ns 1080 -nc 20 -cv wz

    python subsample.py -m uips --path ~/data/cylinder --target drag -ns 1080 -nc 20 --plot

# Compare methods

Used to create a histogram plot which will compare the subsampling distributions of the various methods.

    python compare_methods.py --path ~/data/cylinder --target drag -ns 1080 -nc 20 -cv wz

    python compare_methods.py --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

    python compare_methods.py --dims 3 --dtype sst-binary --path ~/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 20971 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z

# Uniform-in-phase-space testing

See [uips/README.md](uips/README.md)



