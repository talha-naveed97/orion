#!/bin/bash
#SBATCH --ntasks=8                      # 1 core(CPU)
#SBATCH --nodes=1                       # Use 1 node
#SBATCH --job-name=talha_learning_curves           # The name for our job
#SBATCH --mem=10G                        # Default memory per CPU is 3GB.
#SBATCH --partition=smallmem            # We want to use a node with a GPU
#SBATCH --mail-user=talha.naveed@nmbu.no    # Email me when job is done.
#SBATCH --mail-type=ALL                 # Please e-mail me about all messages regarding this job
#SBATCH --output=script-output-%A_%a.out   # The output from the file will be stored in a file with this name
#SBATCH --error=script-output-%A_%a.out    # The error messages will be stored in a file with this name

# Modify the code above to suit your preferences

# Specify which modules to load (We will only load singularity since everything else is there)
module load singularity

## Run the keras-mnist.py file from the singularity container keras_test.sif with access to a GPU
#singularity exec --nv test.sif python test.py
singularity exec talha_orion.sif python script4.py