#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=calculate_constants_12
#SBATCH --mail-user=runxuanj@umich.edu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=tewaria0
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env

# The application(s) to execute along with its input arguments and options:
conda activate my-rdkit-env
python -u calculate_constants.py