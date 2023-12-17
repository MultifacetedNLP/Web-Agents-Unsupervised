python $PROJECT/lamorel/lamorel/src/lamorel_launcher/launch.py \
        --config-path=$PROJECT/experiments/configs \
        --config-name=my_local_config \
        rl_script_args.path=$PROJECT/experiments/train_language_agent.py \
        rl_script_args.saving_path_model=$SCRATCH/storage/models \
        rl_script_args.saving_path_logs=$SCRATCH/storage/logs \
        rl_script_args.run_name=flan_t5_large_only_ppo \
        rl_script_args.environment_args.extra_search_path=$SCRATCH/data/goal_query_predict.json