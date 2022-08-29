#!/bin/bash
#SBATCH --account=rokabe		# username to associate with job
#SBATCH --job-name=radmap_openmc		# a desired name to appear alongside job ID in squeue
#SBATCH --gres=gpu:1 			# number of GPUs (per node)
#SBATCH --time=3-23:00			# time (DD-HH:MM)
#SBATCH --output="slurm/%x_%j.out"		# output file where all text printed to terminal will be stored
					# current format is set to "job-name_jobID.out"
nice python mcsimulation_tetris.py 	# command or script to run; can use 'nvidia-smi' as a test
