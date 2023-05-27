#!/bin/bash

#SBATCH --job-name=Fin_project
#SBATCH --ntasks=6
#SBATCH -o Fin_project.out
#SBATCH -e Fin_project.out
#SBATCH --open-mode truncate
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

python granger_pycc.py
# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n 6 python mpi_job.py