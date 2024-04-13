#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=100G
#SBATCH --time=2:59:00
#SBATCH --output=../output/%j.out
#SBATCH --account=def-y2863che
#SBATCH --mail-user=muhammed.computecanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

cd /home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmsegmentation
python computecanada/submit/save_SIC_SOD_FLOE.py