"""
This file is used to run experiments and calculate performance metrics on the data in realtime. The code is taken from other jupyter notebooks
in the repository, replicating the same performance metric logging in the `/process_data/single_example_data_analysis` notebooks
1. Data is plucked from the `hpc_runs_society_data` folder, and data id is logged
2. The dataset is aggregated on, and the HCVA point and transition point are calculated, and logged
3. The performance of each strategy as described in the `single_example_data_analysis_notebooks` folder is calculated and logged for each strategy
"""

import pandas as pd
import solve
import secrets
import wandb
from datetime import date

from matrices import FormalisationObjects, FormalisationMatrix

# Global variables
personal_data = None
principle_data = None
path = ""

def calculate_rankings(strategies, cons_vals, action_cons_vals) -> List[pd.DataFrame]:
    # Create an agent_principle_data_df to 

    for row in agent_principle_data_df.iterrows():
        index, row_data = row
        egal = row_data['egal']
        util = row_data['util']
        # Normalising means that egal is 0 and util is 1
        normalised_value = egal / (egal + util)
        agent_principle_data_df.at[index, 'normalised_value'] = normalised_value

    # Now each row is normalised, we convert the normalised value into the principle value for each agent, between 1.0 and 10.0
    for row in agent_principle_data_df.iterrows():
        index, row_data = row
        normalised_value = row_data['normalised_value']
        principle_value = normalised_value * 9 + 1
        agent_principle_data_df.at[index, 'principle_value'] = round(principle_value, 2)

    # Now calculate the ranking for of consensus rankings for each principle we are interested in
    for row in agent_principle_data_df.iterrows():
        index, row_data = row
        temp_distances = {}
        for principle, value in consensus_rankings.items():
            agent_principle = row_data['principle_value']
            temp_distances[principle] = abs(agent_principle - value)
        temp_distances = {key: value for key, value in sorted(temp_distances.items(), key=lambda item: item[1])}
        # Place in dataframe for agent
        for i, (principle, distance) in enumerate(temp_distances.items()):
            agent_principle_data_df.at[index, principle + "_rank"] = i + 1
    # Calculate the frequency of each strategy at each rank
    rank_frequencies = agent_principle_data_df[['hcva_rank', 't_rank', 'util_rank', 'egal_rank']].apply(pd.Series.value_counts).fillna(0).astype(int)
    # Sum up the ranks for each principle
    rank_sums = agent_principle_data_df[['hcva_rank', 't_rank', 'util_rank', 'egal_rank']].sum()
    return rank_frequencies, rank_sums


def run_experiment(filename, iteration, contextnum, samplenum) -> None:

    # formalise current personal data to matrices to run aggregation
    P_list, J_list, w, country_dict = FormalisationObjects(
    filename=None, delimiter=',', df=personal_data)
    
    # Note: debug phrase: "Could not find JMatrix" is printed for principle data (there is no action judgement data in our principle value system)
    PP_list, PJ_list, Pw, Pcountry_dict = FormalisationObjects(
        filename=None, delimiter=',', df=principle_data)
    
    # Values for current experiment to save to file
    experiment_cons_vals = []
    experiment_action_cons_vals = []
    experiment_decisions = []
    scores = []

    limit_p_filename = path+"limit_p_"+filename+'_iter_'+str(iteration)+'_context_'+str(contextnum)+'_sample_'+str(samplenum)+'.csv'

    # Solve aggregation for sample here: do all P values in 0.1 steps from 0 to 10
    transition_point, hcva_point, cons_vals, action_cons_vals = solve.aggregate(P_list, 
                                                                                J_list, 
                                                                                w, 
                                                                                country_dict, 
                                                                                PP_list, 
                                                                                PJ_list, 
                                                                                Pw, 
                                                                                Pcountry_dict, 
                                                                                principle_data, 
                                                                                limit_p_filename)
    
    # Reason about points to make decisions for
    # Save data
    strategies = [1, 10, transition_point, hcva_point]
    print("DEBUG: Strategies to be analysed:", strategies)
    for p in strategies:
        # Convert p value to point in list (list is 0.1 increments)
        p = int((p-1)*10)
        # TODO: Place all data analysis here
        decision = solve.make_decision(cons_vals[p], action_cons_vals[p])
    
    rankings_df, ranks_sums = calculate_rankings(strategies=strategies, cons_vals=cons_vals, action_cons_vals=action_cons_vals)
    # TODO: calculate_distances()
    
    return scores, experiment_decisions, experiment_cons_vals, experiment_action_cons_vals, transition_point, hcva_point

########
# Main #
########
if __name__ == '__main__':
    # Columns to be used in aggreagation and analysis
    example_data_names = ['country', 'rel', 'nonrel', 'a_div_rel', 'a_div_nonrel']
    example_principle_names = ['country', 'egal', 'util']

    # Types of societies to be looked at. Each dataset is created to have principles with different overall trends
    society_names = ['util_society', 
                     'egal_society', 
                     'norm_society', 
                     'rand_society']
    societies = ['util_societies', 
                 'egal_societies', 
                 'norm_societies', 
                 'rand_societies']

    timestamp = date.today().strftime('%Y-%m-%d') 

    # Sanity check to stop wandb to stop overrunning
    if wandb.run is not None:
        wandb.finish()
    
    # Iterate over all different society types, to get data for each
    for society in societies.items():
        # Separate experiment for each society type
        wandb.init(project="valuesystemsaggregation",
        config={"society": society,
                "timestamp": timestamp,
                }
        )
        for i in range(1, 50):
            # read in data
            #FOR BLUEPEBBLE: data = pd.read_csv('/user/home/ia23938/ValueSystemsAggregation/data/society_data/'+folder+'/'+name+'_'+str(random)+'.csv')
            data = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/multiple_example_data/'+society+'/'+society+'_'+str(i)+'.csv')

            # Extract values and action judgements for eg. 1
            # need pp, P_1, a_enjoy_camp, a_enjoy_resort, a_budget_camp, a_budget_resort
            personal_data = data[example_data_names]
            principle_data = data[example_principle_names]
            principle_data = data.rename(columns={'egal' : 'rel', 'util' : 'nonrel'})

            experiment_score, decisions, preference_consensus, action_judgement_consensus, transition_point, hcva_point = run_experiment(society, i)

            # log metrics to wandb
            wandb.log({"iteration": iterator, 
                        "hcva_score" : hcva_score,
                        "transition_score" : transition_score,
                        "util_score" :util_score,
                        "egal_score" : egal_score
                        })
            # Clear memory
            try:
                solve.shutdown_julia()
                del personal_data, principle_data
                #gc.collect()
            except Exception as e:
                print("DEBUG: Error in memory clearing, exception:", e)
    
    print("DEBUG: Experiment complete")