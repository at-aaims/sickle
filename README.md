# SICKLE 

SICKLE = Sparse Intelligent data Curation framework for Knowledge-driven Learning Efficiency 

SICKLE is a tool to "separate the wheat from the chaff", that is, to extract 
data with the probabilistically highest information content to improve the 
cost of training large models.

# Subsampling

Login to Frontier

    source /lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate

    OPENBLAS_NUM_THREADS=4 python subsample.py --method maxent --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

# Training

    module load cray-python/3.10.10 rocm

    python -u train-pt-ddp.py --epochs 10 --patience 100 --dims 3 --dtype sst-binary --noseed -ns 10000 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 128 --window 5 --arch transformer

See https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html

# Tests on laptop - random and maxent

    python subsample.py -m random --path ../DataSiftML/data/cylinder --target drag -ns 540
    python subsample.py -m maxent --path ../DataSiftML/data/cylinder --target drag -ns 540 -cv p

# Tests on Frontier

    source '/lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate'

    OPENBLAS_NUM_THREADS=4 python subsample.py maxent --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

# Parallel tests on Frontier

    # OpenFOAM dataset - random and maxent

    srun -n 4 python -u subsample-mpi.py -m random --path ../datasiftml/data --target drag -ns 540
    srun -n 4 python -u subsample-mpi.py -m maxent --path ../datasiftml/data --target drag -ns 540

    # debk dataset
    OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py -m maxent --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

# To compare MaxEnt methods with Phase-space-sampling approach

From the [Phase-space-sampling repo](https://github.com/NREL/Phase-space-sampling), run

    srun -np 4 python tests/main_from_input.py -i input2D

This will output several files. The only one used now is:

    downSampledData_1000_it1.npz

copy this one to sickle repo. Also, copy `combustion2DToDownsampleSmall.npy` to the `data` directory in the sickle repo.

Then run:

    python uips-eval.py

This will output two png images: 

    scatter_maxent_downsampled.png  
    scatter_phasespace_downsampled.png

These two images can be compared to compare the two methods. 

To cluster on 2D, change the following line in algorithms.py from:

    data = cv[timestep, :].reshape(-1, 1)

to:

    data = cv[timestep, :]
