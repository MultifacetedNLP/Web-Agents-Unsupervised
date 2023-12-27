import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_log(dir_):
    """Loads log from a directory and adds it to a list of dataframes."""
    df = pd.read_csv(os.path.join(dir_, 'log.csv'),
                     on_bad_lines='warn')
    if not len(df):
        print("empty df at {}".format(dir_))
        return
    df['model'] = dir_
    return df


def load_logs(root):
    dfs = []
    for root, dirs, files in os.walk(root, followlinks=True):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_log(root))
    return dfs

def plot_average_impl(df, regexps, labels, limits, colors, y_value='return_mean', window=50, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])
    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models, label, color in zip(regexps, model_groups, labels, colors):
        # print("regex: {}".format(regex))
        print(models)
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                                for _, df_model in df_re.groupby('model')]
        for _, df_model in df_re.groupby('model'):
            print(df_model[x_value].max())
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= limits]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pd.concat(parts)
        df_re = df_re.drop('model', axis=1)
        df_agg = df_re.groupby([x_value]).mean()
        # df_max = df_re.groupby([x_value]).max()[y_value]
        # df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]
        std = df_re.groupby([x_value]).std()[y_value]
        # print(std.iloc[-1])
        df_max = values + std
        df_min = values - std

        # pyplot.plot(df_agg.index, values, label='{} SE: {}'.format(label, round(values.sum()/len(values), 3)))
        print(("{} last mean:{} last std: {}").format(label, values.iloc[-1], std.iloc[-1]))
        plt.plot(df_agg.index, values, label=label, color=color)
        # pyplot.plot(df_agg.index, values, label=label)
        plt.fill_between(df_agg.index, df_max, df_min, alpha=0.25, color=color)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])
        print("{} sample efficiency: {}".format(label, values.sum() / len(values)))

"""
def plot_average_impl(df, regex, labels, limits, colors, y_value='return_mean', window=50, agg='mean',
                      x_value='frames'):
    # Plot averages over groups of runs  defined by regular expressions.
    df = df.dropna(subset=[y_value])
    unique_models = df['model'].unique()
    
    model_groups = []
    for reg in regex:
        for m in unique_models:
            if re.match(reg, m):
                model_groups.append(m)
                break

    for regex, model, label, color in zip(regex, model_groups, labels, colors):
        print(model)
        df_re = df[df['model'] == model]
        df_re = df_re[df_re[x_value] <= limits]


        df_re.loc[:, y_value] = df_re[y_value].rolling(window).mean()

        values = df_re[y_value]

        plt.plot(df_re.frames, values, label=label, color=color)

        print("{} sample efficiency: {}".format(label, values.sum() / len(values)))
"""

def plot_average_impl_ax(df, regexps, ax, labels, limits, colors, y_value='return_mean', window=10, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])
    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models, label, color in zip(regexps, model_groups, labels, colors):
        # print("regex: {}".format(regex))
        print(models)
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                                for _, df_model in df_re.groupby('model')]
        for _, df_model in df_re.groupby('model'):
            print(df_model[x_value].max())
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= limits]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pd.concat(parts)
        df_agg = df_re.groupby([x_value]).mean()
        # df_max = df_re.groupby([x_value]).max()[y_value]
        # df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]
        std = df_re.groupby([x_value]).std()[y_value]
        # print(std.iloc[-1])
        df_max = values + std
        df_min = values - std

        # pyplot.plot(df_agg.index, values, label='{} SE: {}'.format(label, round(values.sum()/len(values), 3)))
        print(("{} last mean:{} last std: {}").format(label, values.iloc[-1], std.iloc[-1]))
        ax.plot(df_agg.index, values, label=label, color=color)
        # pyplot.plot(df_agg.index, values, label=label)
        ax.fill_between(df_agg.index, df_max, df_min, alpha=0.25, color=color)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])
        print("{} sample efficiency: {}".format(label, values.sum() / len(values)))


dfs = load_logs('/u/spa-d4/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/storage/logs_lcc')
df = pd.concat(dfs, sort=True)

def plot_average_and_success_rate_average(**kwargs):
    """Plot averages and success rate averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 6))
    new_kwargs = {key: value for key, value in kwargs.items() if key not in ['average_labels', 'average_colors', 'success_rate_average_labels', 'success_rate_average_colors']}
    plot_average_impl(y_value='return_mean', labels=kwargs['average_labels'], colors=kwargs['average_colors'], **new_kwargs)
    plot_average_impl(y_value='success_rate', labels=kwargs['success_rate_average_labels'], colors=kwargs['success_rate_average_colors'],**new_kwargs)
    # plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11}, bbox_to_anchor=(1.1, 1.1))
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("PPO Training Process", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    # plt.figure(figsize=(8, 6), dpi=100)
    plt.show()

def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='return_mean', *args, **kwargs)
    # plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11}, bbox_to_anchor=(1.1, 1.1))
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("Average Score", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    # plt.figure(figsize=(8, 6), dpi=100)
    plt.show()


def plot_success_rate_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='success_rate', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("Average Success Rate", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.show()


def plot_entropy_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='entropy', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("Average Entropy", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("Average Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_policy_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='policy_loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("Average Policy Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_value_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='value_loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("Average Value Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

def plot_grad_norm_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='grad_norm', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Steps", fontsize=15)

    plt.title("Average Grad Norm", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

# #######################MTRL##############################################################


"""
regexs = ['.*flan_T5_large_2_observations_category_.*',
          '.*flan_T5_large_2_observations_all_nbr_env_16_nbr_obs_2_.*',
          '.*flan_T5_large_only_ppo_nbr_env_16_nbr_obs_2_.*']
average_labels = ['Score of Unsupervised D.A.', 'Score of  SL + PPO', 'Score of  RL (PPO)']
success_rate_average_labels = ['Success Rate of Unsupervised D.A.', 'Success Rate of SL + PPO', 'Success Rate of RL (PPO)']
limits = 1000000
average_colors = ['#003399', '#339933', '#CC3333']
success_rate_average_colors = ['#3399FF', '#66FF66', '#FF6666']
plot_average_and_success_rate_average(df=df, regexps=regexs, average_labels=average_labels, success_rate_average_labels=success_rate_average_labels, 
                                      limits=limits, average_colors=average_colors, success_rate_average_colors=success_rate_average_colors)
# plot_average(df, regexs, labels, limits, colors)
# plot_success_rate_average(df, regexs, labels, limits, colors)
"""


"""
regexs = ['.*flan_T5_large_1_observation_all_nbr_env_16_nbr_obs_1_.*',
          '.*flan_T5_large_2_observations_all_nbr_env_16_nbr_obs_2_.*']
labels = ['One Observation', 'Two Observations']
average_labels = ['Score of One Observation', 'Score of Two Observations']
success_rate_average_labels = ['Success Rate of One Observation', 'Success Rate of Two Observations']
limits = 500000
average_colors = ['#003399', '#880000']
success_rate_average_colors = ['#3399FF', '#FF6666']
plot_average_and_success_rate_average(df=df, regexps=regexs, average_labels=average_labels, success_rate_average_labels=success_rate_average_labels, 
                                      limits=limits, average_colors=average_colors, success_rate_average_colors=success_rate_average_colors)
# plot_average(df, regexs, labels, limits, colors)
# plot_success_rate_average(df, regexs, labels, limits, colors)
"""

"""
regexs = ['.*t5_large_only_ppo_nbr_env_16_nbr_obs_2_.*',
          '.*flan_T5_large_only_ppo_nbr_env_16_nbr_obs_2_.*']
average_labels = ['Score of T5 Large', 'Score of Flan T5 Large']
success_rate_average_labels = ['Success Rate of T5 Large', 'Success Rate of Flan T5 Large']
limits = 500000
average_colors = ['#003399', '#880000']
success_rate_average_colors = ['#3399FF', '#FF6666']
plot_average_and_success_rate_average(df=df, regexps=regexs, average_labels=average_labels, success_rate_average_labels=success_rate_average_labels, 
                                      limits=limits, average_colors=average_colors, success_rate_average_colors=success_rate_average_colors)
# plot_average(df, regexs, labels, limits, colors)
# plot_success_rate_average(df, regexs, labels, limits, colors)
"""



regexs = ['.*t5_large_2_observations_category_.*',
          '.*flan_T5_large_2_observations_category_.*']
average_labels = ['Score of T5 Large with D.A.', 'Score of Flan T5 Large with D.A.']
success_rate_average_labels = ['Success Rate of T5 Large with D.A.', 'Success Rate of Flan T5 Large with D.A.']
limits = 500000
average_colors = ['#003399', '#880000']
success_rate_average_colors = ['#3399FF', '#FF6666']
plot_average_and_success_rate_average(df=df, regexps=regexs, average_labels=average_labels, success_rate_average_labels=success_rate_average_labels, 
                                      limits=limits, average_colors=average_colors, success_rate_average_colors=success_rate_average_colors)
# plot_average(df, regexs, labels, limits, colors)
# plot_success_rate_average(df, regexs, labels, limits, colors)
