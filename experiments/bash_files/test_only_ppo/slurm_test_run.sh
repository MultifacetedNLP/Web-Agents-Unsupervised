#! /bin/bash
#SBATCH --time=3-00:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=Flan_T5_webshop # Job name
#SBATCH -e Flan_T5_webshop_slurm-%j.err # Error file for this job.
#SBATCH -o Flan_T5_webshop_slurm-%j.out # Output file for this job.
#SBATCH --partition=V4V32_CAS40M192_L # Partition/queue to run the job in. (REQUIRED)
#SBATCH -A gol_msi290_uksr # Project allocation account name (REQUIRED)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

python $PROJECT/lamorel/lamorel/src/lamorel_launcher/launch.py \
        --config-path=$PROJECT/experiments/configs \
        --config-name=my_local_test_config \
        rl_script_args.path=$PROJECT/experiments/test_webshop.py \
        rl_script_args.id_expe=flan_t5_large_2_observations_only_ppo_1000000_steps \
        rl_script_args.saving_path_model=$SCRATCH/storage/models \
        rl_script_args.saving_path_logs=$SCRATCH/storage/logs