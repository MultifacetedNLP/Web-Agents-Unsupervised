python /u/spa-d2/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/lamorel/lamorel/src/lamorel_launcher/launch.py \
        --config-path=/u/spa-d2/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/experiments/configs \
        --config-name=my_local_test_config \
        rl_script_args.path=/u/spa-d2/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/experiments/test_webshop.py \
        rl_script_args.id_expe=flan_t5_large_2_observations_only_ppo_1000000_steps \
        rl_script_args.saving_path_model=$SCRATCH/storage/models \
        rl_script_args.saving_path_logs=$SCRATCH/storage/logs