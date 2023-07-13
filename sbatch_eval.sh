#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --error=log/%j.err
#SBATCH  --mem=40G
#SBATCH  --cpus-per-task=4

source /scratch_net/biwidl204/agassol/conda/etc/profile.d/conda.sh
conda activate volsdf_cu11

set -o errexit
# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

cd code
# Run the python script
python -u evaluation/eval.py --scan_id 65 --gpu auto --timestamp 2023_04_05_19_45_40 --eval_rendering

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0