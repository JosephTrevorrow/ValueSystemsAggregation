"""
This file creates graphs tracking satisfaction over time for each agent in the simulation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

#####################
# Utility Functions #
#####################

def unpack_data(filename: str):
    df = pd.read_csv(filename)
    return df

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

def split_by_context(data: pd.DataFrame):
    ## This function splits a DataFrame by context, such that you have 3 seperate DataFrames for each context

    df_dict = {}
    unique_contexts = data['context'].unique()
    # Split the DataFrame by Context
    for context in unique_contexts:
        df_dict[f'df_context_{context}'] = data[data['context'] == context].reset_index(drop=True)
        
    return df_dict

def split_into_size_groups(data: pd.DataFrame):
    """
    This function splits up the dataframe into 3 corresponding to the sizes of 25x4, 10x10, and 4x25 for analysis by group

    Contexts' 0-24 are the first 25x4, contexts' 25-34 are the first 10x10, contexts' 35-38 are the first 4x25. then the context value resets
       to 0 and this continues
    """

    df_list = []
    size_groups = [(0, 25), (25, 35), (35, 39)]
    for start, end in size_groups:
        df_group = data[(data['context'] >= start) & (data['context'] < end)].reset_index(drop=True)
        df_list.append(df_group)

    return df_list

############################
# Agent Specific Functions #
############################

def find_best_worst_case_divergence_agent(data: pd.DataFrame, filter: int):
    """
    finds the agent that is worst off in terms of divergence
    """
    worst_agent = None
    best_agent = None
    worst_sat = 0
    best_sat = 999
    df_dict = {}
    unique_agent_values = data['agent'].unique()
    
    # Filter out specific P values
    data = data[data['p_value'] == filter]

    # Split the DataFrame by agent value
    for agent in unique_agent_values:
        df_dict[f'df_p_{agent}'] = data[data['agent'] == agent].reset_index(drop=True)
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].sort_values(by=['agent', 'context']).reset_index(drop=True)
    for key in df_dict:
        # Calculate cumulative sum of Satisfaction for each Agent
        df_dict[key]['Cumulative_Satisfaction'] = df_dict[key].groupby('agent')['satisfaction'].cumsum()
    for key in df_dict:
        # Reset the Context for each Agent group
        df_dict[key]['context'] = df_dict[key].groupby('agent').cumcount()
    
    # Find the worst agent
    for key in df_dict:
        df = df_dict[key]
        if df['Cumulative_Satisfaction'].iloc[-1] > worst_sat:
            worst_agent = df
            worst_sat = df['Cumulative_Satisfaction'].iloc[-1]

    # Find the best case P
    for key in df_dict:
        df = df_dict[key]
        if df['Cumulative_Satisfaction'].iloc[-1] < best_sat:
            best_agent = df
            best_sat = df['Cumulative_Satisfaction'].iloc[-1]

    return worst_agent, best_agent

def find_decision_cumulative_divergence(data: pd.DataFrame):
    """
    This function calcluates the cumulative divergence as:
    - For each agent in a deicison, find their difference from the consensus. Now sum these differences for each agent per decision.
    - This is the cumulative divergence for that decision
    """
    df_dict = {}
    unique_p_values = data['p_value'].unique()
    # Split the DataFrame by P_Value
    for p_value in unique_p_values:
        df_dict[f'df_p_{p_value}'] = data[data['p_value'] == p_value].reset_index(drop=True)
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].sort_values(by=['agent', 'context']).reset_index(drop=True)
    results_dict = {}
    # Iterate over each DataFrame in df_dict
    for key, df in df_dict.items():
        # Group by continuous segments where 'context' is unchanging
        df['group'] = (df['context'] != df['context'].shift()).cumsum()
        # Sum the 'satisfaction' values for each group
        grouped = df.groupby('group').agg({
            'context': 'first',
            'satisfaction': 'sum'
        }).reset_index(drop=True)
        
        # Add the p_value to the grouped DataFrame
        grouped['p_value'] = df['p_value'].iloc[0]
        
        # Store the result in the dictionary with p_value as the key
        results_dict[f'df_p_{grouped["p_value"].iloc[0]}'] = grouped

    return results_dict

def find_agent_cumulative_divergence(data: pd.DataFrame):
    """
    This function calculates the cumulative divergence as:
    - For each agent, calculuate its cumulative sum of satisfaction over time
    - This is the cumulative divergence for that agent
    """
    df_dict = {}
    unique_p_values = data['p_value'].unique()
    colours = ['red', 'green', 'blue', 'orange']
    # Split the DataFrame by P_Value
    for p_value in unique_p_values:
        df_dict[f'df_p_{p_value}'] = data[data['p_value'] == p_value].reset_index(drop=True)
    for key in df_dict:
        # Calculate cumulative sum of Satisfaction for each Agent
        df_dict[key]['Cumulative_Satisfaction'] = df_dict[key].groupby('agent')['satisfaction'].cumsum()

    for key in df_dict:
        # Reset the Context for each Agent group
        df_dict[key]['context'] = df_dict[key].groupby('agent').cumcount()

    return df_dict



def find_best_worst_off_agents_overall(data: pd.DataFrame):
    return

def find_best_worst_off_agents_over_time(data: pd.DataFrame, society_name: str):
    """
    For every context, find the agent that is the worst off in terms of divergence, and store as a list
    see if one agent (or a small group of agents in a minority) are consistently the worst off
    """
    df_dict = {}
    unique_p_values = data['p_value'].unique()
    # Split the DataFrame by P_Value
    for p_value in unique_p_values:
        df_dict[f'df_p_{p_value}'] = data[data['p_value'] == p_value].reset_index(drop=True)
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].sort_values(by=['agent', 'context']).reset_index(drop=True)
    results_dict = {}
    # Iterate over each DataFrame in df_dict
    for key, df in df_dict.items():
        # Group by 'context'
        grouped = df.groupby('context')
        max_min_list = []
        for name, group in grouped:
            max_row = group.loc[group['satisfaction'].idxmax()]
            min_row = group.loc[group['satisfaction'].idxmin()]
            max_min_list.append({
                'context': name,
                'agent_max': max_row['agent'],
                'satisfaction_max': max_row['satisfaction'],
                'agent_min': min_row['agent'],
                'satisfaction_min': min_row['satisfaction']
            })
        results_dict[key] = pd.DataFrame(max_min_list)

    # Save results_dict to a .csv file and find any common agents
    for key, df in results_dict.items():
        df.to_csv(f"{society_name}_{key}_results.csv", index=False)
        # Find common agents
        common_worst_agents = df['agent_max'].mode()
        common_best_agents = df['agent_min'].mode()
        common_worst_agents.to_csv(f"{society_name}_{key}_common_worst_agents.csv", index=False)
        common_best_agents.to_csv(f"{society_name}_{key}_common_best_agents.csv", index=False)
        # print(f"Common worst agents for {key}: {common_worst_agents}")

    return results_dict


######################
# Plotting Functions #
######################

def plot_violin_best_worst(worst_agent: pd.DataFrame, best_agent: pd.DataFrame, plot_savename: str, name:str, pname:str):
    # TODO: UNFINISHED
    """
    This function plots cumulative divergence for each strategy in the data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """
    plt.style.use("ggplot")
    colours = ['red', 'blue']
    dfs = [worst_agent, best_agent]
    plt.figure(figsize=(6, 6))
    labels = ['Best off agent', 'Worst off agent']
    # Loop through each DataFrame in df_dict
    for (df, colour, label) in zip(dfs, colours, labels):
        # Plot the line for this dataframe
        plt.plot(df['Cumulative_Satisfaction'], color=colour, label=label)
    # Add labels and title
    plt.xlabel('Context')
    plt.ylabel('Cumulative Divergence')
    plt.title(f' Cumulative Divergence for worst and best case\n {name} society')
    plt.legend(title='Legend')
    plt.ylim(0, 100)  # Set the y-axis limits
    plt.xlim(0,320)
    plt.grid(True)
    
    plt.savefig(plot_savename+title)

    difference_in_satisfaction = dfs[0]['Cumulative_Satisfaction'].iloc[-1] - dfs[1]['Cumulative_Satisfaction'].iloc[-1]

    with open("best_worst_divergence.csv", 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, pname, difference_in_satisfaction])

    for (df,label) in zip(dfs, labels):
        df.to_csv(plot_savename+name+label+'.csv')


def plot_worst_best_cumulative_divergence(worst_agent: pd.DataFrame, best_agent: pd.DataFrame, title: str, plot_savename: str, name: str, pname: str):
    """
    This function plots cumulative divergence for each strategy in the data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """

    plt.style.use("ggplot")
    colours = ['red', 'blue']
    dfs = [worst_agent, best_agent]
    plt.figure(figsize=(6, 6))
    labels = ['Best off agent', 'Worst off agent']
    # Loop through each DataFrame in df_dict
    for (df, colour, label) in zip(dfs, colours, labels):
        # Plot the line for this dataframe
        plt.plot(df['Cumulative_Satisfaction'], color=colour, label=label)
        
        # Plot the spread (shaded area)
        #plt.fill_between(mean_cum_satisfaction.index,
        #                 min_cum_satisfaction,
        #                 max_cum_satisfaction,
        #                 color=colour, alpha=0.1)

    # Add labels and title
    plt.xlabel('Context')
    plt.ylabel('Cumulative Divergence')
    plt.title(f' Cumulative Divergence for worst and best case\n {name} society')
    plt.legend(title='Legend')
    plt.ylim(0, 100)  # Set the y-axis limits
    plt.xlim(0,320)
    plt.grid(True)
    
    plt.savefig(plot_savename+title)

    difference_in_satisfaction = dfs[0]['Cumulative_Satisfaction'].iloc[-1] - dfs[1]['Cumulative_Satisfaction'].iloc[-1]

    with open("best_worst_divergence.csv", 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, pname, difference_in_satisfaction])

    for (df,label) in zip(dfs, labels):
        df.to_csv(plot_savename+name+label+'.csv')

    # TODO: Find the agents actual value systems by looking at the logs for experiments


def plot_decision_cumulative_divergence(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots cumulative divergence for each decision
    """
    plt.style.use("ggplot")
    colours = ['red', 'green', 'blue', 'orange']
    
    df_dict = find_decision_cumulative_divergence(data)
    
    plt.figure(figsize=(6, 6))
    labels = ['1 (Utilitarian)', '10 (Egalitarian)', 'Transition Point', 'HCVA Point']
    # Loop through each DataFrame in df_dict
    for (df, colour, label) in zip(df_dict, colours, labels):
        # Plot the line for this dataframe
        plt.plot(df_dict[df]['satisfaction'], color=colour, label=label)

    # Add labels and title
    plt.xlabel('Context')
    plt.ylabel('Total Divergence for Decision')
    # plt.title(f'Worst case Cumulative Divergence for Different P Values\n {name} society')
    plt.legend(title='P Values')
    plt.ylim(0, 12)  # Set the y-axis limits
    plt.xlim(0,320)
    plt.grid(True)    
    
    plt.savefig(plot_savename+title)

def plot_cumulative_divergence(data: pd.DataFrame, title: str, plot_savename: str, name: str):
    """
    This function plots cumulative divergence for each strategy in the data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """

    plt.style.use("ggplot")
    colours = ['red', 'green', 'blue', 'orange']
    
    df_dict = find_agent_cumulative_divergence(data)
    
    plt.figure(figsize=(6, 6))
    labels = ['1 (Utilitarian)', '10 (Egalitarian)', 'Transition Point', 'HCVA Point']
    # Loop through each DataFrame in df_dict
    for (key, colour, label) in zip(df_dict, colours, labels):
        # Calculate the mean, min, and max of Cumulative Satisfaction for each Context across all Agents
        mean_cum_satisfaction = df_dict[key].groupby('context')['Cumulative_Satisfaction'].mean()
        min_cum_satisfaction = df_dict[key].groupby('context')['Cumulative_Satisfaction'].min()
        max_cum_satisfaction = df_dict[key].groupby('context')['Cumulative_Satisfaction'].max()

        # Plot the line for this dataframe
        plt.plot(mean_cum_satisfaction, color=colour, label=label)
        
        # Plot the spread (shaded area)
        #plt.fill_between(mean_cum_satisfaction.index,
        #                 min_cum_satisfaction,
        #                 max_cum_satisfaction,
        #                 color=colour, alpha=0.1)

    # Add labels and title
    plt.xlabel('Context')
    plt.ylabel('Worst case Cumulative Divergence')
    #plt.title(f'Worst case Cumulative Divergence for Different P Values\n {name} society')
    plt.legend(title='P Values')
    plt.ylim(0, 100)  # Set the y-axis limits
    plt.xlim(200,320)
    plt.grid(True)    
    
    plt.savefig(plot_savename+title)

    #import tikzplotlib
    #tikzplotlib.save("text.tex")

    # TODO: Return the exact difference at the end, and what agent was the worst (their value system)

def plot_boxplot_residuals(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots a boxplot of the satisfaction residuals for each test in the data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """
    satisfaction_df = data.groupby(['agent', 'p_value'], as_index=False).agg({'satisfaction': 'sum'})
    satisfaction_df['p_value'] = satisfaction_df['p_value'].replace({0: 'Utilitarian', 1: 'Egalitarian', 2: 'Transition Point', 3: 'HCVA'})

    plt.style.use("ggplot")
    plt.figure(figsize=(6, 6))
    sns.boxplot(x='p_value', y='satisfaction', data=satisfaction_df, width=0.3, whis=3, color="grey")

    #plt.title(title)
    plt.xlabel('Strategy')
    plt.ylabel('Total Agent Divergence')
    plt.ylim(45 ,120)  # Set the y-axis limits
    savename = plot_savename+title
    plt.savefig(savename)

    # TODO: Return the exact difference at the end, and what agent was the worst (their value system)


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



def plot_satisfaction_of_minority(data: pd.DataFrame, title: str, plot_savename: str):

    return 

def cumulative_divergence_of_sizes(data: pd.DataFrame, title: str, plot_savename: str):
    """
    This function plots 3 different graphs for cumulative divergence for each size group within a society to show any differences in group
    - This function ignores P value differences
    """
    data = split_into_size_groups(data=data)
    colours = ['red', 'green', 'blue', 'orange']
    labels = ['Small (4)', 'Medium (10)', 'Large (25)']

    for df in data:
        # Calculate cumulative sum of Satisfaction for each Agent
        df['Cumulative_Satisfaction'] = df.groupby('agent')['satisfaction'].cumsum()
        df.groupby('agent').cumcount()
        
    plt.figure(figsize=(12, 8))

    # Loop through each DataFrame in df_dict
    for (key, colour, label) in zip(data, colours, labels):
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
    plt.ylabel('Mean Cumulative Divergence')
    plt.title(f'Mean Cumulative Divergence over Time for Different P Values\n {name} society')
    plt.legend(title='P Values')
    plt.ylim(0, 100)  # Set the y-axis limits
    plt.xlim(0,320)
    plt.grid(True)
    plt.savefig(plot_savename+title)

    return


#############################
# below is runner functions #
#############################

def violins_best_worst_agents():
    print("DEBUG: Unpacking data")
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_runs/experiment_results_2024-09-27/"
    results_filename = {'egal': "egal_society/egal_societyegal_society.csv", 'norm': "norm_society/norm_societynorm_society.csv", "util": "util_society/util_societyutil_society.csv", "random": "rand_society/rand_societyrand_society.csv"}
    filters = {"egal": 0, "util": 1, "t": 2, "HCVA": 3}
    for pname, p_val in filters.items():
        for name, filename in results_filename.items():
            data = unpack_data(results_path + filename)
            worst_agent, best_agent = find_best_worst_case_divergence_agent(data, filter=p_val)
            plot_violin_best_worst(worst_agent, best_agent, f"Cumulative Agent Satisfaction Over Time for {name} society and P value {pname}", plot_savename, name, pname)

def best_worst_case():
    print("DEBUG: Unpacking data")
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_runs/experiment_results_2024-09-27/"
    results_filename = {'egal': "egal_society/egal_societyegal_society.csv", 'norm': "norm_society/norm_societynorm_society.csv", "util": "util_society/util_societyutil_society.csv", "random": "rand_society/rand_societyrand_society.csv"}
    filters = {"egal": 0, "util": 1, "t": 2, "HCVA": 3}
    for pname, p_val in filters.items():
        for name, filename in results_filename.items():
            data = unpack_data(results_path + filename)
            worst_agent, best_agent = find_best_worst_case_divergence_agent(data, filter=p_val)
            #plot_worst_best_cumulative_divergence(worst_agent, best_agent, f"Cumulative Agent Satisfaction Over Time for {name} society and P value {pname}", plot_savename, name, pname)
            find_best_worst_off_agents_over_time(data, name)

def boxplots_and_cumulative():
    print("DEBUG: Unpacking data")
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_runs/experiment_results_2024-09-27/"
    results_filename = {'egal': "egal_society/egal_societyegal_society.csv", 'norm': "norm_society/norm_societynorm_society.csv", "util": "util_society/util_societyutil_society.csv", "random": "rand_society/rand_societyrand_society.csv"}
    for name, filename in results_filename.items():
        data = unpack_data(results_path + filename)
        plot_boxplot_residuals(data, f"Total Agent Divergence for {name} society", plot_savename)
        plot_decision_cumulative_divergence(data, f"Decision Divergence Over Time for {name} society", plot_savename)
        plot_cumulative_divergence(data, f"Cumulative Divergence Over Time for {name} society", plot_savename, name)

def t_points():
    results_filename = {'egal_dist/egal_dist_hcva_points.csv': "egal_dist/egal_dist_t_points.csv", 'normal_dist/norm_dist_hcva_points.csv': "normal_dist/norm_dist_t_points.csv", "util_dist/util_dist_hcva_points.csv": "util_dist/util_dist_t_points.csv", "random_dist/rand_dist_hcva_points.csv": "random_dist/rand_dist_t_points.csv"}
    
    results_filename = {"egal_society/egal_societyegal_society_hcva_points.csv": "egal_society/egal_societyegal_society_t_points.csv", "norm_society/norm_societynorm_society_hcva_points.csv": "norm_society/norm_societynorm_society_t_points.csv", "util_society/util_societyutil_society_hcva_points.csv": "util_society/util_societyutil_society_t_points.csv", "rand_society/rand_societyrand_society_hcva_points.csv": "rand_society/rand_societyrand_society_t_points.csv"}
    savenames = ['egalitarian society', 'normal society', 'utilitarian society', 'random society']
    for (hcva, t_point), savename in zip(results_filename.items(), savenames):
        hcva_data = unpack_data(results_path + hcva)
        t_data = unpack_data(results_path + t_point)
        plot_transition_and_hcva_points(folders, hcva_data, t_data, f"Transition and HCVA Points for {savename}",savename)

def decisiveness():
    savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_runs/experiment_results_2024-09-27/"
    results_filename = ["egal_society/egal_societyegal_society_DECISIONS.csv", "normal_society/normal_societynormal_society_DECISIONS.csv", "util_society/util_societyutil_society_DECISIONS.csv", "random_society/random_societyrandom_society_DECISIONS.csv"]
    savenames = ['decs_egalitarian_society', 'decs_normal_society', 'decs_utilitarian society', 'decs_random_society']
    for filename, savename in zip(results_filename, savenames):
        data = unpack_data(results_path + filename)
        plot_decisiveness(data, f"Decisiveness for {savename}", savename)

def fairness():
    print("DEBUG: Unpacking data")
    #plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_plots/"
    #results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_results/"
    results_filename = {'egal': "egal_society/egal_societyegal_society.csv", 'norm': "norm_society/norm_societynorm_society.csv", "util": "util_society/util_societyutil_society.csv", "random": "rand_society/rand_societyrand_society.csv"}
    for name, filename in results_filename.items():
        data = unpack_data(results_path + filename)
        plot_fairness_thresholds(data, f"Fairness Thresholds for {name} society", plot_savename+name)

if __name__ == "__main__":
    
    #boxplots_and_cumulatives()
    plot_savename = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_plots/"
    results_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/bluepebble_runs/experiment_results_2024-09-27/"
    # Assuming you just have a folder name now e.g. 'experiment_results_v2/random_dist'
    folders = 'experiment_results_v2/random_dist'
    boxplots_and_cumulative()