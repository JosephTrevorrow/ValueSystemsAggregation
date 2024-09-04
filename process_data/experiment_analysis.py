"""
This file creates graphs tracking satisfaction over time for each agent in the simulation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

def plot_cumulative_satisfaction(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots the average satisfaction for each test in the data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """

    df_dict = {}
    unique_p_values = data['p_value'].unique()
    colours = ['red', 'green', 'blue', 'orange']
    # Split the DataFrame by P_Value
    for p_value in unique_p_values:
        df_dict[f'df_p_{p_value}'] = data[data['p_value'] == p_value].reset_index(drop=True)
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].sort_values(by=['agent', 'context']).reset_index(drop=True)
    for key in df_dict:
        # Calculate cumulative sum of Satisfaction for each Agent
        df_dict[key]['Cumulative_Satisfaction'] = df_dict[key].groupby('agent')['satisfaction'].cumsum()

    for key in df_dict:
        # Reset the Context for each Agent group
        df_dict[key]['context'] = df_dict[key].groupby('agent').cumcount()
        
    plt.figure(figsize=(12, 8))
    labels = ['1', '10', 't', 'p']
    # Loop through each DataFrame in df_dict
    for (key, colour, label) in zip(df_dict, colours, labels):
        # Calculate the mean, min, and max of Cumulative Satisfaction for each Context across all Agents
        mean_cum_satisfaction = df_dict[key].groupby('context')['Cumulative_Satisfaction'].mean()
        min_cum_satisfaction = df_dict[key].groupby('context')['Cumulative_Satisfaction'].min()
        max_cum_satisfaction = df_dict[key].groupby('context')['Cumulative_Satisfaction'].max()

        # Plot the line for this dataframe
        plt.plot(mean_cum_satisfaction, color=colour, label=label)
                # Plot the spread (shaded area)
        plt.fill_between(mean_cum_satisfaction.index,
                         min_cum_satisfaction,
                         max_cum_satisfaction,
                         color=colour, alpha=0.1)

    # Add labels and title
    plt.xlabel('Context')
    plt.ylabel('Mean Cumulative Satisfaction')
    plt.title('Mean Cumulative Satisfaction over Time for Different P_Values')
    plt.legend(title='P_Values')
    plt.ylim(0, 20)  # Set the y-axis limits
    plt.grid(True)
    plt.savefig(plot_savename+title)

def plot_boxplot_residuals(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots a boxplot of the satisfaction residuals for each test in the data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """
    satisfaction_df = data.groupby(['agent', 'p_value'], as_index=False).agg({'satisfaction': 'sum'})
    satisfaction_df['p_value'] = satisfaction_df['p_value'].replace({0: '1', 1: '10', 2: 't', 3: 'p'})

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='p_value', y='satisfaction', data=satisfaction_df, width=0.3, whis=3)

    plt.title(title)
    plt.xlabel('P Value')
    plt.ylabel('Satisfaction')
    plt.ylim(0, 20)  # Set the y-axis limits
    savename = plot_savename+title
    plt.savefig(savename)

def plot_transition_and_hcva_points(folders: str, hcva_points: str, t_points: str, title: str, plot_savename: str):
    """
    This function plots the transition and hcva points for each  on a scatter plot.
    """ 
    plt.figure(figsize=(10, 5))
    # Calculate the difference between hcva_points and t_points
    smaller_length = min(len(hcva_points), len(t_points))
    hcva_points = hcva_points[:smaller_length]
    t_points = t_points[:smaller_length]
    difference = np.array(t_points) - np.array(hcva_points)
    # Plot the difference line
    plt.plot(difference, color="green", label="Difference")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Difference")
    plt.savefig(plot_savename)

def plot_limit_p_data(data: pd.DataFrame, title: str, plot_savenme: str):
    """
    This function plots the limit P data (-t True).
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """

    plt.figure(figsize=(10, 5))
    plt.plot(data["p"], data["Dist_p"], label="$||P^{(1)}_S-P^{(P)}_S||_p$")
    plt.plot(data["p"], data["Dist_inf"], label="$||P^{(\infty)}_S-P^{(P)}_S||_p$")
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"], where=(data["Dist_p"] >= data["Dist_inf"]) & (data["Dist_inf"] >= 0.05), color="blue", alpha=0.3, label="Egalitarian Zone")
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"], where=data["Dist_p"] <= data["Dist_inf"], color="green", alpha=0.3, label="Utilitarian Zone")

    #fill dark blue in fully egalitarian zone, $epsilon=0.05$. dist_inf is less than $epsilon$
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"], where=data["Dist_inf"] <= 0.05, color="darkblue", alpha=0.3, label="Fully Egalitarian Zone")
    # Mark the transition point
    transition_point = data.loc[data["Dist_p"] >= data["Dist_inf"], ["p", "Dist_p"]].iloc[0]
    plt.plot(transition_point["p"], transition_point["Dist_p"], "ro", label="Transition Point", markersize=10)
    plt.title(title)
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.legend()
    savename = plot_savename+title
    plt.savefig(savename)

def npfloat64remover(x):
    x = x.replace('np.float64(', '')
    x = x.replace(')', '')
    x = x.replace(']', '')
    x = x.replace('[', '')
    x = x.replace(' ', '')
    return x

def decisionsplitter(x):
    y, z = x.split(',')
    return [float(y), float(z)]

def plot_decisiveness(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots the decisiveness measure (the difference in the strength of two actions) for each $p$ value as a boxplot for each $p$
    """
    means = {}
    p_vals = ['1', '10', 't', 'HCVA']
    for i in range(0, 4):
        plt.figure(figsize=(10, 5))
        # split the difference
        filtered = data[data['p'] == i]
        filtered['decision'] = filtered['decision'].apply(npfloat64remover)
        filtered['decisionsplit']=filtered['decision'].apply(decisionsplitter)
        
        filtered['decision_diff'] = filtered['decisionsplit'].apply(lambda x: abs(x[0]-x[1]))
        filtered = filtered.sort_values(by='decision_diff', ascending=False).reset_index(drop=True)

        mean_dec_diff = filtered['decision_diff'].mean()
        means[plot_savename+'p'+p_vals[i]] = mean_dec_diff
        plt.axhline(mean_dec_diff, color='red', linestyle='--', linewidth=2, label=f'Mean Value: {mean_dec_diff:.2f}')
        plt.ylim(0, 1)  # Set the y-axis limits
        #plt.xlim(-1, 1)  # Set the y-axis limits

        plt.bar(filtered['context'], filtered['decision_diff'], color="green", label="Difference")
        plt.xlabel("Context")
        plt.ylabel("Decisiveness")
        plt.title("Decisiveness for "+p_vals[i])
        plt.savefig(plot_savename+'p'+p_vals[i])
    
    # Save means to a CSV file
    means_df = pd.DataFrame.from_dict(means, orient='index', columns=['Mean_Decisiveness'])
    means_df.to_csv(plot_savename + 'decs_means.csv')

    #plt.scatter(data['x'], data['y'])
    # Calculate mean x and y
    #filtered['x'] = filtered['decisionsplit'].apply(lambda x: x[0])
    #filtered['y'] = filtered['decisionsplit'].apply(lambda x: x[1])
    #mean_x = data['x'].mean()
    #mean_y = data['y'].mean()
    #plt.scatter(mean_x, mean_y, color='red', marker='x', s=100, label='Mean')


def plot_fairness_percentage(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots fairness defined as the average percentage of group members whose personal value system results in the same decision 
    being made as the consensus value system. We calculate the individual values and return a table.
    """
    


def plot_fairness_thresholds(data: pd.DataFrame, title: str, plot_savename: str):
    """
    Fairness defined as the satisfaction of an agent with a certain decision being below some value 0.05 (as a satisfaction of 0 corresponds to maximum 
    satisfaction)
    """
    df_dict = {}
    unique_p_values = data['p_value'].unique()
    # Split the DataFrame by P_Value
    for p_value in unique_p_values:
        df_dict[f'df_p_{p_value}'] = data[data['p_value'] == p_value].reset_index(drop=True)
   
    vals = {}
    p_vals = ['1', '10', 't', 'HCVA']
    i = 0
    for df in df_dict:
        num_below_threshold = len(df_dict[df][df_dict[df]['satisfaction'] < 0.05])
        vals[p_vals[i]] = num_below_threshold
        i += 1
    # Save means to a CSV file
    vals_df = pd.DataFrame.from_dict(vals, orient='index', columns=['Num_Below_Threshold'])
    vals_df.to_csv(plot_savename +'fairness_thresholds.csv')



def unpack_data(filename: str):
    df = pd.read_csv(filename)
    return df

def boxplots_and_cumulatives():
    print("DEBUG: Unpacking data")
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_results/"
    results_filename = {'egal': "egal_dist/egal_dist.csv", 'norm': "normal_dist/norm_dist.csv", "util": "util_dist/util_dist.csv", "random": "random_dist/rand_dist.csv"}
    for name, filename in results_filename.items():
        data = unpack_data(results_path + filename)
        plot_boxplot_residuals(data, f"Agent Satisfaction Over Time for {name} society", plot_savename)
        plot_cumulative_satisfaction(data, f"Cumulative Agent Satisfaction Over Time for {name} society", plot_savename)
    
def t_points():
    results_filename = {'egal_dist/egal_dist_HCVA_POINTS.csv': "egal_dist/egal_dist_T_POINTS.csv", 'normal_dist/norm_dist_HCVA_POINTS.csv': "normal_dist/norm_dist_T_POINTS.csv", "util_dist/util_dist_HCVA_POINTS.csv": "util_dist/util_dist_T_POINTS.csv", "random_dist/rand_dist_HCVA_POINTS.csv": "random_dist/rand_dist_T_POINTS.csv"}
    savenames = ['egalitarian societyz', 'normal society', 'utilitarian society', 'random society']
    for (hcva, t_point), savename in zip(results_filename.items(), savenames):
        hcva_data = unpack_data(results_path + hcva)
        t_data = unpack_data(results_path + t_point)
        plot_transition_and_hcva_points(folders, hcva_data, t_data, f"Transition and HCVA Points for {savename}",savename)

def decisiveness():
    results_filename = ["egal_dist/egal_dist_DECISIONS.csv", "normal_dist/norm_dist_DECISIONS.csv", "util_dist/util_dist_DECISIONS.csv", "random_dist/rand_dist_DECISIONS.csv"]
    savenames = ['decs_egalitarian_society', 'decs_normal_society', 'decs_utilitarian society', 'decs_random_society']
    for filename, savename in zip(results_filename, savenames):
        data = unpack_data(results_path + filename)
        plot_decisiveness(data, f"Decisiveness for {savename}", savename)

def fairness():
    print("DEBUG: Unpacking data")
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_results/"
    results_filename = {'egal': "egal_dist/egal_dist.csv", 'norm': "normal_dist/norm_dist.csv", "util": "util_dist/util_dist.csv", "random": "random_dist/rand_dist.csv"}
    for name, filename in results_filename.items():
        data = unpack_data(results_path + filename)
        plot_fairness_percentage(data, f"Fairness Percentage for {name} society", plot_savename+name)
        plot_fairness_thresholds(data, f"Fairness Thresholds for {name} society", plot_savename+name)

if __name__ == "__main__":
    
    #boxplots_and_cumulatives()
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_results/"
    # Assuming you just have a folder name now e.g. 'experiment_results_v2/random_dist'
    folders = 'experiment_results_v2/random_dist'
    fairness()