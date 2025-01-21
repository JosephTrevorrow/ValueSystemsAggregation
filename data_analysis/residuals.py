import numpy as np
import pandas as pd

#def calculate_unnormalised_residual(agents, cons):

def calculate_normalised_residual(agents, cons):
    # Normalise all data between 0-1
    for column in agents.columns:
        min_val = agents[column].min()
        max_val = agents[column].max()
        if column == 'country':
            continue
        agents[column] = (agents[column] - min_val) / (max_val - min_val)
    comparison_data = ['Rel_div_p', 'Nonrel_div_p', 'Rel-Nonrel', 'Nonrel-Rel']
    cons = cons[comparison_data]
    cons = cons.astype(float)
    for column in cons.columns:
        if column == 'p':
            continue
        min_val = agents[column].min()
        max_val = agents[column].max()
        cons[column] = (cons[column] - min_val) / (max_val - min_val)
    # Now calculate the ranking for of consensus rankings for each principle we are interested in \ 'p_normalised'
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
    agents = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/08-01-2025-agent-data.csv')
    agents.rename(columns={'principle_value': 'p', 'a_div_rel' : 'Rel_div_p', 'a_div_nonrel': 'Nonrel_div_p'}, inplace=True)
    agents.drop(columns=['country'], inplace=True)
    agents = agents.astype(float)
    cons_action = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/08-01-2025-actions.csv')
    cons_prefs = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/08-01-2025-preferences.csv')
    cons = pd.merge(cons_action, cons_prefs, on='p')
    ## TODO: make sure relevant_cons are exhaustive
    relevant_cons = {'Utilitarian': 1.0, 'Egalitarian': 10.0, 'transition_point': 1.3, 'Bottom_25': 1.3, "Top_25": 2.5, "Exreme_Egal": 3.4,
             "Extreme_Util": 1.1, "General_Support_75": 4.1, "General_Opposition": 6.0,
             "General_Support_50": 1.1, "General_Opposition_50": 6.0,
             "ESS_Data": 2.5}
    # Calculate residuals
    for name, con in relevant_cons.items():
        margin_of_error = 1e-9
        con_details = cons[np.isclose(cons['p'], con, atol=margin_of_error)]
        temp_file = calculate_normalised_residual(agents, con_details)
        temp_df = pd.DataFrame(temp_file, columns=['residual'])
        temp_df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/' + name + '.csv', index=False)