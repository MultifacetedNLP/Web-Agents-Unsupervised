#! /bin/bash
#SBATCH --time=20:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=Flan_T5_webshop # Job name
#SBATCH -e Flan_T5_webshop_slurm-%j.err # Error file for this job.
#SBATCH -o Flan_T5_webshop_slurm-%j.out # Output file for this job.
#SBATCH --partition=P4V12_SKY32M192_L # Partition/queue to run the job in. (REQUIRED)
#SBATCH -A gol_msi290_uksr # Project allocation account name (REQUIRED)
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

#SBATCH --array=1-2

module load python/3.8.2
conda activate ./envs/

python /u/spa-d2/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/lamorel/lamorel/src/lamorel_launcher/launch.py \
        --config-path=/u/spa-d2/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/experiments/configs \
        --config-name=my_local_config \
        rl_script_args.path=/u/spa-d2/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/experiments/train_language_agent.py