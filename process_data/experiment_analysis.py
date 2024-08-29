"""
This file creates graphs tracking satisfaction over time for each agent in the simulation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

def plot_agent_satisfaction(data: pd.DataFrame, title: str):
    """
    This function plots the satisfaction data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """


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
    savename = plot_savename+title
    plt.savefig(savename)

def plot_transition_and_hcva_points(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots the transition and hcva points for each  on a scatter plot.
    """ 
    plt.figure(figsize=(10, 5))
    plt.style.use("science")
    

def plot_data(data: pd.DataFrame, title: str, plot_savenme: str):
    """
    This function plots the limit P data (-t True).
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """

    plt.figure(figsize=(10, 5))
    plt.style.use("science")
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


def unpack_data(filename: str):
    df = pd.read_csv(filename)
    return df


if __name__ == "__main__":
    print("DEBUG: Unpacking data")
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/results_v1/plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/results_v1/"
    results_filename = {'egal': "egalsoc_10iteration_3diffagent.csv", 'norm': "normsoc_10iteration_3diffagent.csv"}
    for name, filename in results_filename.items():
        data = unpack_data(results_path + filename)
        plot_boxplot_residuals(data, f"Agent Satisfaction Over Time for {name} society", plot_savename)
        plot_cumulative_satisfaction(data, f"Cumulative Agent Satisfaction Over Time for {name} society", plot_savename)
    
    # Assuming you just have a folder name now e.g. 'experiment_results_v2/random_dist'
    folders = 'experiment_results_v2/random_dist'
    t_points = 'random_distrandom_dist_t_points.csv'
    hcva_points = 'random_distrandom_dist_hcva_points.csv'
    data = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/'+folders+filename+'.csv')