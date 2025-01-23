import numpy as np
import pandas as pd

def calculate_normalised_residual(agents, cons):
    # Now calculate the ranking for of consensus rankings for each principle we are interested in \ 'p_normalised'
    comparison_data = ['Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']
    temp_distances = []
    for agent_row in agents.iterrows():
        for cons_row in cons.iterrows():
            for column in comparison_data:
                agent_value = agent_row[1][column]
                consensus_value = cons_row[1][column]
                abs_difference = abs(agent_value - consensus_value)
            temp_distances.append(abs_difference)
    return temp_distances

if __name__ == "__main__":
    # Import agents data and consensus data
    agents = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/22-01-2025-agent-data.csv')
    agents.rename(columns={'rel': 'Rel-Nonrel', 'nonrel': 'Nonrel-Rel', 'a_div_rel' : 'Rel_div_p', 'a_div_nonrel': 'Nonrel_div_p'}, inplace=True)
    agents.drop(columns=['country'], inplace=True)
    agents = agents.astype(float)
    agents = agents[['Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']]
    cons_action = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/22-01-2025-actions.csv')
    cons_prefs = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/22-01-2025-preferences.csv')
    cons = pd.merge(cons_action, cons_prefs, on='p')

    # Import MSE data
    mse_data_path = "/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/means_and_salas_molina_ps/slm_consensus.csv"
    mse_df = pd.read_csv(mse_data_path)
    # Convert all columns other than 'p' in mse_df to float
    for column in mse_df.columns:
        if column != 'p' and column != 'series_name':
            mse_df[column] = mse_df[column].astype(float)
    mse_df = mse_df[['p','Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']]

    # Normalise all data for consensus between 0-1
    consensus_data_df = cons.astype(float)
    consensus_data_df = consensus_data_df[['p','Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']]
    for column in consensus_data_df.columns:
        min_val = consensus_data_df[column].min()
        max_val = consensus_data_df[column].max()
        if column != 'p':
            consensus_data_df[column] = (consensus_data_df[column] - min_val) / (max_val - min_val)

    # Normalise mse values between 0-1 using CONSENSUS and then add these to the bottom of the consensus data
    for column in mse_df.columns:
        if column != 'p' and column != 'series_name':
            min_val = consensus_data_df[column].min()
            max_val = consensus_data_df[column].max()
            mse_df[column] = (mse_df[column] - min_val) / (max_val - min_val)
    
    consensus_data_df = pd.concat([consensus_data_df, mse_df], ignore_index=True)

    # Normalise all agents data between 0-1 using CONSENSUS
    for column in agents.columns:
        min_val = consensus_data_df[column].min()
        max_val = consensus_data_df[column].max()
        if column == 'country':
            continue
        agents[column] = (agents[column] - min_val) / (max_val - min_val)

    #####
    relevant_cons = {'Bottom_25': 1.3, "Top_25": 2.5, "Exreme_Egal": 3.4,
             "Extreme_Util": 1.1, "General_Support_75": 4.1, "General_Opposition_75": 6.0,
             "General_Support_50": 1.1, "General_Opposition_50": 6.0,
             "ESS_Data": 2.5,'Utilitarian': 1.0,
                 'Egalitarian': 10.0,
                 'Transition': 2.2,'slm_Bottom_25': 994,    
             "slm_Top_25": 990, 

             "slm_Exreme_Egal": 992,
             "slm_Extreme_Util": 999, 

             "slm_General_Support_75": 995, 
             "slm_General_Opposition_75": 996,

             "slm_General_Support_50": 997, 
             "slm_General_Opposition_50": 991,
             "slm_ESS_Data": 989}
    #####
    
    # Calculate residuals for each p value in relevant_cons and save to files
    for name, con in relevant_cons.items():
        margin_of_error = 1e-9
        con_details = consensus_data_df[np.isclose(consensus_data_df['p'], con, atol=margin_of_error)]
        print("DEBUG: CON DETAILS IS: ",con_details)
        temp_file = calculate_normalised_residual(agents, con_details)
        temp_df = pd.DataFrame(temp_file, columns=['residual'])
        temp_df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/normalised_residuals/principles_included_' + name + '.csv', index=False)