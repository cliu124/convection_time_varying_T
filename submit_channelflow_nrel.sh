#!/bin/bash
#SBATCH --account=ducnguyen1410
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=30-00:00:00
###SBATCH --mail-user=chang_liu@uconn.edu      # Destination email address
#SBATCH --mail-type=END                       # Event(s) that triggers email notification
#SBATCH --job-name=channelflow_job
#SBATCH --output=channelflow_output_%j
export SLURM_EXPORT_ENV=ALL
#export I_MPI_FABRICS=shm,tcp

##the slurm number to restart simulation... This need full state to be stored.

##### sbatch -A ducnguyen1410 submit_channelflow_nrel.sh
####sbatch -A uconnfluent submit_dedalus_nrel

SUBMITDIR=$SLURM_SUBMIT_DIR
WORKDIR=/scratch/ducnguyen1410/channelflow_$SLURM_JOB_ID
mkdir -p "$WORKDIR" &&  cd "$WORKDIR" || exit -1

###which mpicxx
### make builduconn

cd "$SUBMITDIR" && cp channelflow_output_$SLURM_JOB_ID "$WORKDIR"

