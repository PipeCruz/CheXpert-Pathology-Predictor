#!/bin/bash

#SBATCH --nodes=1

#SBATCH --partition=gpu

# Name of job in SLURM queue
#SBATCH --job-name=cs156b_poptart_job1

# Output and error log locations (captures stdout and stderr)
#SBATCH --output=/home/ezhang3/output/%j.out
#SBATCH --error=/home/ezhang3/output/%j.err

# Account to charge this computation to. THIS LINE IS ESSENTIAL.
#SBATCH -A CS156b

# Estimated time this job will take.
#SBATCH -t 2:00:00

# Total number of concurrent srun tasks. Most people will not need this.
#SBATCH --ntasks=1

# Number of CPU threads for each task as defined above.
#SBATCH --cpus-per-task=12

# Total amount of system RAM for all tasks. Be decently generous with this.
#SBATCH --mem-per-cpu=8G

# Request a single Tesla P100 GPU.
#SBATCH --gres=gpu:1

# Send status update to email
#SBATCH --mail-user=ezhang3@caltech.edu

# Enable email notifications for changes to job status
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

### FIXME the file name should be changed each time
cd /home/ezhang3/156b/
/home/fcruzfal/anaconda3/bin/python3 modelalexnet_ckpt.py --file_name solution_submission_alexnet.csv --num_gpus 1
