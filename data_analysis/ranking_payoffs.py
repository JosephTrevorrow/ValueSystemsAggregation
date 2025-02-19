import pandas as pd

def get_rankings(agent_data_df, consensus_data_df, hcva, hcva_name, mse_df, mse):
    agent_data_df = agent_data_df[['country','Rel-Nonrel','Nonrel-Rel', 'a_div_rel', 'a_div_nonrel']]
    consensus_data_df = consensus_data_df[['p','Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']]
    agent_data_df.rename(columns={'a_div_rel' : 'Rel_div_p', 'a_div_nonrel': 'Nonrel_div_p'}, inplace=True)
    mse_df = mse_df[['p','Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']]

    # Normalise all data between 0-1
    consensus_data_df = consensus_data_df.astype(float)
    for column in consensus_data_df.columns:
        min_val = consensus_data_df[column].min()
        max_val = consensus_data_df[column].max()
        if column == 'p':
            consensus_data_df['p_normalised'] = consensus_data_df['p']
            column = 'p_normalised'
        consensus_data_df[column] = (consensus_data_df[column] - min_val) / (max_val - min_val)

    # Normalise mse values
    for column in mse_df.columns:
        if column != 'p' and column != 'series_name':
            print("DEBUG:",column)
            min_val = consensus_data_df[column].min()
            max_val = consensus_data_df[column].max()
            mse_df[column] = (mse_df[column] - min_val) / (max_val - min_val)
    consensus_data_df = pd.concat([consensus_data_df, mse_df], ignore_index=True)
    #consensus_data_df.loc[len(consensus_data_df)]
    print(consensus_data_df.tail())

    # Filter consensus_data_df to keep only rows with p values of interest (BASELINES)
    relevant_p_values = [1.0, 2.2, hcva, 10.0, mse]
    tolerance = 1e-5
    print("DEBUG: MSE IS: ", mse)
    print("DEBUG: HCVA IS: ", hcva) 
    consensus_data_df = consensus_data_df[consensus_data_df['p'].apply(lambda x: any(abs(x - val) < tolerance for val in relevant_p_values))]

    # Now the same for agents, but work from the unnormalised consensus data
    for column in agent_data_df.columns:
        if column == 'country' or column == 'p_normalised':
            continue
        min_val = agent_data_df[column].min()
        max_val = agent_data_df[column].max()
        if column == 'p':
            agent_data_df['p_normalised'] = agent_data_df['p']
            column = 'p_normalised'
        agent_data_df[column] = (agent_data_df[column] - min_val) / (max_val - min_val)
    # Now calculate the ranking for of consensus rankings for each principle we are interested in \ 'p_normalised'
    comparison_data = ['Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']
    for agent_row in agent_data_df.iterrows():
        temp_strategy_distances = {}
        temp_distances = []
        for consensus_row in consensus_data_df.iterrows():
            for column in comparison_data:
                agent_value = agent_row[1][column]
                consensus_value = consensus_row[1][column]
                abs_difference = abs(agent_value - consensus_value)
                temp_distances.append(abs_difference)
            #print(temp_distances)
            temp_strategy_distances[consensus_row[1]['p']] = sum(temp_distances)
            temp_distances = []
        strategy_temp_distances = {key: value for key, value in sorted(temp_strategy_distances.items(), key=lambda item: item[1])}
        # Place in dataframe for agent
        for i, (principle, distance) in enumerate(strategy_temp_distances.items()):
            #print(f"Principle: {principle}, Distance: {distance}")
            principle = round(principle, 1)
            agent_data_df.at[agent_row[0], principle] = i + 1
    agent_data_df.rename(columns={1.0 : 'util_rank', hcva : hcva_name, 2.2: 't_rank', 10.0 : 'egal_rank', mse : 'mse_rank'}, inplace=True)
    return agent_data_df

def latex_rank_sums(agent_data_dfs, hcva_names):
    # Initialize a dictionary to store the rank sums for each principle
    rank_sums_dict = {name: {'hcva_rank': 0, 't_rank': 0, 'util_rank': 0, 'egal_rank': 0, 'mse_rank': 0} for name in hcva_names}
    #print(agent_data_dfs[0])
    # Sum up the ranks for each principle
    for agent_data_df, hcva_name in zip(agent_data_dfs, hcva_names):
        print(hcva_name)
        rank_sums = agent_data_df[[hcva_name, 't_rank', 'util_rank', 'egal_rank', 'mse_rank']].sum()
        rank_sums_dict[hcva_name]['hcva_rank'] = int(rank_sums[hcva_name])
        rank_sums_dict[hcva_name]['t_rank'] = int(rank_sums['t_rank'])
        rank_sums_dict[hcva_name]['util_rank'] = int(rank_sums['util_rank'])
        rank_sums_dict[hcva_name]['egal_rank'] = int(rank_sums['egal_rank'])
        rank_sums_dict[hcva_name]['mse_rank'] = int(rank_sums['mse_rank'])

    # Create a DataFrame from the rank sums dictionary
    rank_sums_df = pd.DataFrame(rank_sums_dict).T

    # Create a LaTeX table
    latex_table = rank_sums_df.to_latex(index=True, header=['hcva rank', 't rank', 'util rank', 'egal rank', 'mse_rank'], caption='Sum of rankings given to each strategy by agents (Lower is better).')
    with open('mse_rank_sums_table.tex', 'w') as f:
        f.write(latex_table)
def latex_rank_fptp(agent_data_dfs, hcva_names):
    # Initialize a dictionary to store the rank sums for each principle
    rank_sums_dict = {name: {'hcva_rank': 0, 't_rank': 0, 'util_rank': 0, 'egal_rank': 0, 'mse_rank': 0} for name in hcva_names}
    # Sum up the ranks for each principle where the value is 1.0
    for agent_data_df, hcva_name in zip(agent_data_dfs, hcva_names):
        rank_sums = agent_data_df[[hcva_name, 't_rank', 'util_rank', 'egal_rank', 'mse_rank']].apply(lambda x: x[x == 1.0].sum())
        rank_sums_dict[hcva_name]['hcva_rank'] = int(rank_sums[hcva_name])
        rank_sums_dict[hcva_name]['t_rank'] = int(rank_sums['t_rank'])
        rank_sums_dict[hcva_name]['util_rank'] = int(rank_sums['util_rank'])
        rank_sums_dict[hcva_name]['egal_rank'] = int(rank_sums['egal_rank'])
        rank_sums_dict[hcva_name]['mse_rank'] = int(rank_sums['mse_rank'])

    # Create a DataFrame from the rank sums dictionary
    rank_sums_df = pd.DataFrame(rank_sums_dict).T

    # Create a LaTeX table
    latex_table = rank_sums_df.to_latex(index=True, header=['hcva rank', 't rank', 'util rank', 'egal rank', 'mse_rank'], caption='Sum of rankings given to each strategy by agents (Lower is better).')
    with open('mse_rank_fptp_table.tex', 'w') as f:
        f.write(latex_table)


def latex_single_rankings(agent_data_df, hcva_name):
    # Calculate the frequency of each strategy at each rank
    #print(agent_data_df)
    rank_frequencies = agent_data_df[[hcva_name, 't_rank', 'util_rank', 'egal_rank', 'mse_rank']].apply(pd.Series.value_counts).fillna(0).astype(int)

    # Create a LaTeX table
    latex_rank_frequencies = rank_frequencies.T.to_latex()

    print(latex_rank_frequencies)

def latex_borda_counts(agent_data_dfs, hcva_names):
    # Initialize a dictionary to store the Borda scores for each principle
    borda_scores_dict = {name: {'hcva_borda': 0, 't_borda': 0, 'util_borda': 0, 'egal_borda': 0, 'mse_borda': 0} for name in hcva_names}
    # Assign Borda points to each rank
    borda_points = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    for agent_data_df, hcva_name in zip(agent_data_dfs, hcva_names):
        # Sum the Borda points for each principle
        for _, row in agent_data_df.iterrows():
            borda_scores_dict[hcva_name]['util_borda'] += borda_points[row['util_rank']]
            borda_scores_dict[hcva_name]['hcva_borda'] += borda_points[row[hcva_name]]
            borda_scores_dict[hcva_name]['t_borda'] += borda_points[row['t_rank']]
            borda_scores_dict[hcva_name]['egal_borda'] += borda_points[row['egal_rank']]
            borda_scores_dict[hcva_name]['mse_borda'] += borda_points[row['mse_rank']]

    # Create a DataFrame from the Borda scores dictionary
    borda_scores_df = pd.DataFrame(borda_scores_dict).T

    # Create a LaTeX table
    latex_borda_scores = borda_scores_df.to_latex(index=True, header=['hcva borda', 't borda', 'util borda', 'egal borda', 'mse borda'], caption='Borda scores for each strategy (Higher is better).')
    with open('mse_borda_scores_table.tex', 'w') as f:
        f.write(latex_borda_scores)
    
if __name__ == '__main__':
    agent_data_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/22-01-2025-agent-data.csv"
    agent_data_df = pd.read_csv(agent_data_path)
    agent_data_df.rename(columns={'rel': 'Rel-Nonrel', 'nonrel': 'Nonrel-Rel'}, inplace=True)

    consensus_data_path_pref = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/22-01-2025-actions.csv"
    consensus_data_path_act = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/22-01-2025-preferences.csv"
    temp_pref = pd.read_csv(consensus_data_path_pref)
    temp_act = pd.read_csv(consensus_data_path_act)
    consensus_data_df = pd.merge(temp_pref, temp_act, on='p')

    #####
    hcvas = {'Bottom_25': 1.3, "Top_25": 2.5, "Exreme_Egal": 3.4,
             "Extreme_Util": 1.1, "General_Support_75": 4.1, "General_Opposition_75": 6.0,
             "General_Support_50": 1.1, "General_Opposition_50": 6.0,
             "ESS_Data": 2.5}
    #####
    #####
    mse = {'Bottom_25': 994,    
             "Top_25": 990, 

             "Exreme_Egal": 992,
             "Extreme_Util": 999, 

             "General_Support_75": 995, 
             "General_Opposition_75": 996,

             "General_Support_50": 997, 
             "General_Opposition_50": 991,
             "ESS_Data": 989}
    #####

    mse_data_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/means_and_salas_molina_ps/slm_consensus.csv"
    mse_df = pd.read_csv(mse_data_path)
    # Convert all columns other than 'p' in mse_df to float
    for column in mse_df.columns:
        if column != 'p' and column != 'series_name':
            mse_df[column] = mse_df[column].astype(float)
    #print(mse_df)

    # Get all df's
    agent_data_dfs = {}
    for (hcva_name, hcva), (_, mse_value) in zip(hcvas.items(), mse.items()):
        temp_agent_data_df = get_rankings(agent_data_df, consensus_data_df, hcva, hcva_name, mse_df, mse_value)
        agent_data_dfs[hcva_name] = temp_agent_data_df

    # Print Borda Counts and Rankings
    #latex_rank_sums(list(agent_data_dfs.values()), list(agent_data_dfs.keys()))
    #latex_borda_counts(list(agent_data_dfs.values()), list(agent_data_dfs.keys()))
    #latex_single_rankings(agent_data_dfs['ESS_Data'], 'ESS_Data')
    latex_rank_fptp(list(agent_data_dfs.values()), list(agent_data_dfs.keys()))


