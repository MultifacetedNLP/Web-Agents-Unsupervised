#! /bin/bash
#SBATCH --time=3-00:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=Flan_T5_webshop # Job name
#SBATCH -e Flan_T5_webshop_slurm-%j.err # Error file for this job.
#SBATCH -o Flan_T5_webshop_slurm-%j.out # Output file for this job.
#SBATCH --partition=??? # Partition/queue to run the job in. (REQUIRED)
#SBATCH -A ??? # Project allocation account name (REQUIRED)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1


python $PROJECT/experiments/test_webshop.py \
        --config-path=$PROJECT/experiments/configs \
        --config-name=my_local_test_drrn_config