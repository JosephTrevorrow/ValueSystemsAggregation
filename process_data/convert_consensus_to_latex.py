"""
This file will format a .csv that contains consensus value systems generated from solve.py
to a latex table printout to be used in your document
"""

import pandas as pd
import csv
import numpy as np

def convert_consensus_to_latex(action_csv_file, preference_csv_file, columns_to_fill):
    """
    In the following format:
    \begin{table*}[!htb]
    % \scriptsize
    \centering
    \caption{convert_consensus_to_latex}
    \begin{tabular}{lllllllll}
    \toprule
    P & $P_S[enjoy, budget]$ & $P_S[budget, enjoy]$ & $a^S_{enjoy}(camp)$ & $a^S_{enjoy}(resort)$ & $a^S_{budget}(camp)$ & $a^S_{budget}(resprt)$ & Value pref. \\ 
    \midrule
    1  & 0.544 & 0.456 & 0.256&0.782&0.559&-0.441 & $enjoy > budget$\\
    2  & 0.525 & 0.475 & 0.275&0.775&0.600&-0.400 & $enjoy > budget$\\
    \textbf{2.5}     & \textbf{0.516} & \textbf{0.483} & \textbf{0.284} & \textbf{0.772} & \textbf{0.620} & \textbf{-0.380} & \textbf{$enjoy > budget$}\\
    $\infty$ (10) & 0.500 & 0.500 &0.230&0.756&0.650&-0.350& $enjoy = budget$ \\
    \bottomrule
    \end{tabular}
    \label{Tab:personalAgg}
    \end{table*}
    """
    # Read the csv files
    data_1 = pd.read_csv(action_csv_file)
    data_2 = pd.read_csv(preference_csv_file)
    data = pd.merge(data_1, data_2, on='p')
    data = data[columns_to_fill]

    data = generate_decisions_column(data)
    print(data)

    # Round the 'P' columns to 1 dp and all others to 3 dp
    data['p'] = data['p'].apply(lambda x: round(x, 1))
    data['Rel-Nonrel'] = data['Rel-Nonrel'].apply(lambda x: round(x, 3))
    data['Nonrel-Rel'] = data['Nonrel-Rel'].apply(lambda x: round(x, 3))
    data['Rel_div_p'] = data['Rel_div_p'].apply(lambda x: round(x, 3))
    data['Nonrel_div_p'] = data['Nonrel_div_p'].apply(lambda x: round(x, 3))

    # Get the column names
    columns = data.columns
    # Get the number of columns
    num_columns = len(columns)
    # Get the number of rows
    num_rows = len(data)
    # Open the latex table
    latex_table = "\\begin{table*}[!htb]\n"
    latex_table += "% \\scriptsize\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{convert_consensus_to_latex}\n"
    latex_table += "\\begin{tabular}{"
    # Add the columns
    for i in range(num_columns):
        if i == 0:
            latex_table += "l"
        else:
            latex_table += "l"
    latex_table += "}\n"
    latex_table += "\\toprule\n"
    # Add the column names
    for i in range(num_columns):
        if i == 0:
            latex_table += columns[i]
        else:
            latex_table += " & " + columns[i]
    latex_table += " \\\\ \n"
    latex_table += "\\midrule\n"
    # Add the data
    for i in range(num_rows):
        for j in range(num_columns):
            if j == 0:
                latex_table += str(data.iloc[i][j])
            else:
                latex_table += " & " + str(data.iloc[i][j])
        latex_table += " \\\\ \n"
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{Tab:personalAgg}\n"
    latex_table += "\\end{table*}\n"
    # Print the latex table
    print(latex_table)

def convert_agent_data_to_latex(agent_csv_file, columns_to_fill):
    """
    \begin{table*}[!htb]
    % \small
    \centering
    \caption{convert_agent_data_to_latex func}
    \begin{tabular}{@{}lllllllll@{}}
    \toprule
    Country & $P_i[Tr, He]$ & $P_i[He, Tr]$ & $a^i_{Tr}(basinc)$ & $a^i_{He}(basinc)$ & $PriP_i[Egal, Util]$ & $PriP_i[Util, Egal]$ \\
    \midrule
    ES & 0.08 & 0.92 & 0.3   & -0.30 & 0.78 & 0.22 \\
    IT & 0.12 & 0.88 & 0.23  & -0.23 & 0.69 & 0.31 \\
    PT & 0.16 & 0.84 & 0.32  & -0.31 & 0.82 & 0.18 \\
    \bottomrule
    \end{tabular}%
    \label{Tab:exampleAgentValueSystems}
    \end{table*}
    """
    # read the csv file
    data = pd.read_csv(agent_csv_file)
    data = data[columns_to_fill]


    # Round the numeric columns to 3 dp, skip the 'country' column
    for column in columns_to_fill:
        if column != 'country':
            data[column] = data[column].apply(lambda x: round(x, 3))
    # Get the column names
    columns = data.columns
    # Get the number of columns
    num_columns = len(columns)
    # Get the number of rows
    num_rows = len(data)
    # Open the latex table
    latex_table = "\\begin{table*}[!htb]\n"
    latex_table += "% \\small\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{convert_agent_data_to_latex}\n"
    latex_table += "\\begin{tabular}{"
    # Add the columns
    for i in range(num_columns):
        if i == 0:
            latex_table += "l"
        else:
            latex_table += "l"
    latex_table += "}\n"
    latex_table += "\\toprule\n"
    # Add the column names
    for i in range(num_columns):
        if i == 0:
            latex_table += columns[i]
        else:
            latex_table += " & " + columns[i]
    latex_table += " \\\\ \n"
    latex_table += "\\midrule\n"
    # Add the data
    for i in range(num_rows):
        for j in range(num_columns):
            if j == 0:
                latex_table += str(data.iloc[i][j])
            else:
                latex_table += " & " + str(data.iloc[i][j])
        latex_table += " \\\\ \n"
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{Tab:convert_agent_data_to_latex}\n"
    latex_table += "\\end{table*}\n"
    # Print the latex table
    print(latex_table)

def generate_decisions_column(dataframe):
    """
    Generate a new column with the decision made by the agent and its value preference
    INPUT: dataframe (pandas dataframe)
    Return: dataframe with the new columns
    """
    for row in dataframe.iterrows():
        dataframe['decision'] = dataframe.apply(lambda row: make_decision([row['Rel-Nonrel'], row['Nonrel-Rel'], row['Rel_div_p'], row['Nonrel_div_p']]), axis=1)
        dataframe['preference'] = np.where(dataframe['decision'] > 0, 'For Scheme', 'Against Scheme')

    return dataframe

def make_decision(row) -> str:
    """
    This function computes a decision where teh preferecns and actions
    are given as a row in the format [rel-nonrel, nonrel-rel,'Rel_div_p', 'Nonrel_div_p']
    """
    return (row[0] * row[2]) + (row[1] * row[3])


if __name__ == '__main__':
    action_filename = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/14-11-results-factor-2.5-5.0.csv'
    preferences_filename = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/14-11-results-factor-2.5-5.0-prefs.csv'
    agent_csv_file = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/processed_data_one_action_ess.csv_with_factor_2.5_5.0.csv'

    columns_to_fill = ['p', 'Rel-Nonrel', 'Nonrel-Rel', 'Rel_div_p', 'Nonrel_div_p' ]
    convert_consensus_to_latex(action_filename, preferences_filename, columns_to_fill)

    #columns_to_fill = ['country', 'rel', 'nonrel', 'a_div_rel', 'a_div_nonrel']
    #convert_agent_data_to_latex(agent_csv_file, columns_to_fill)