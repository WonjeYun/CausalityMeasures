#!/bin/bash

#SBATCH --job-name=Question_2_a
#SBATCH --ntasks=10
#SBATCH -o Question_2_a.out
#SBATCH -e Question_2_a_error.out
#SBATCH --open-mode truncate
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

python granger_pycc.py
# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n 10 python mpi_job.py