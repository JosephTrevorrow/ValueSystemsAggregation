"""
This file will format a .csv that contains consensus value systems generated from solve.py
to a latex table printout to be used in your document
"""

import pandas as pd
import csv
import numpy as np

def convert_consensus_to_latex(action_csv_file, preference_csv_file, columns_to_fill):
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
    action_filename = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/results/02-01-2025-actions-a-2.5-p-5.0.csv'
    preferences_filename = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/results/02-01-2025-preferences-a-2.5-p-5.0.csv'
    #agent_csv_file = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/processed_data_one_action_ess.csv_with_factor_2.5_5.0.csv'

    columns_to_fill = ['p', 'Rel-Nonrel', 'Nonrel-Rel', 'Rel_div_p', 'Nonrel_div_p' ]
    convert_consensus_to_latex(action_filename, preferences_filename, columns_to_fill)

    #columns_to_fill = ['country', 'rel', 'nonrel', 'a_div_rel', 'a_div_nonrel']
    #convert_agent_data_to_latex(agent_csv_file, columns_to_fill)