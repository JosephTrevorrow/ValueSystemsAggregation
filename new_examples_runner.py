"""
This file is used to run experiments and calculate performance metrics on the data in realtime. The code is taken from other jupyter notebooks
in the repository, replicating the same performance metric logging in the `/process_data/single_example_data_analysis` notebooks
1. Data is plucked from the `hpc_runs_society_data` folder, and data id is logged
2. The dataset is aggregated on, and the HCVA point and transition point are calculated, and logged
3. The performance of each strategy as described in the `single_example_data_analysis_notebooks` folder is calculated and logged for each strategy
"""

# TODO: Check if the wandb recording is correct, is each soc type being recorded correctly?

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

def calculate_rankings(strategies,
        cons_vals,
        action_cons_vals,
        hcva_point,
        transition_point,) -> list[pd.DataFrame]:
    """
    This function wil use the consensus values to calculate the rankings of each strategy for each agent
    """
    # Create an agent_principle_data_df to 
    print("DEBUG: Calculating rankings for each strategy")

    util_data = {'consensus': cons_vals[1], 'action_consensus': action_cons_vals[1]}
    egal_data = {'consensus': cons_vals[10], 'action_consensus': action_cons_vals[10]}
    transition_point_index = int((transition_point)*10)
    transition_data = {'consensus': cons_vals[transition_point_index], 'action_consensus': action_cons_vals[transition_point_index]}
    hcva_point_index = int((hcva_point)*10)
    hcva_data = {'consensus': cons_vals[hcva_point_index], 'action_consensus': action_cons_vals[hcva_point_index]}

    # Create a dataframe of all of the data, from the cons_vals (preferences) and 
    # action_cons_values (actions) with each column being "p, Rel_div_p, Nonrel_div_p, 
    # Rel-Nonrel, Nonrel-Rel". Each row increases p by 0.1
    data = []
    for p in range(0, 101):
        p = int(p)
        print(p)
        data.append({
            'p': p/10,
            'Rel_div_p': action_cons_vals[p][1],
            'Nonrel_div_p': action_cons_vals[p][2],
            'Rel-Nonrel': cons_vals[p][1],
            'Nonrel-Rel': cons_vals[p][2]
        })

    df = pd.DataFrame(data)
    print(df.head())
    df.to_csv('rankingstest.csv')

    #return rank_frequencies, rank_sums
    return

def run_experiment(filename, iteration, societynum) -> None:

    # formalise current personal data to matrices to run aggregation
    P_list, J_list, w, country_dict = FormalisationObjects(
    filename=None, delimiter=',', df=personal_data)
    
    # Note: debug phrase: "Could not find JMatrix" is printed for principle data (there is no action judgement data in our principle value system)
    print(principle_data.head())
    PP_list, PJ_list, Pw, Pcountry_dict = FormalisationObjects(
        filename=None, delimiter=',', df=principle_data)
    
    # Values for current experiment to save to file
    experiment_cons_vals = []
    experiment_action_cons_vals = []
    experiment_decisions = []
    scores = []

    # Solve aggregation for sample here: do all P values in 0.1 steps from 0 to 10
    transition_point, hcva_point, cons_vals, action_cons_vals = solve.aggregate(P_list, 
                                                                                J_list, 
                                                                                w, 
                                                                                country_dict, 
                                                                                PP_list, 
                                                                                PJ_list, 
                                                                                Pw, 
                                                                                Pcountry_dict, 
                                                                                principle_data)
    
    # Reason about points to make decisions for
    # Save data
    strategies = [1, 10, transition_point, hcva_point]

    # Decision calculation removed

    rankings_df, ranks_sums = calculate_rankings(
        strategies=strategies, 
        cons_vals=cons_vals, 
        action_cons_vals=action_cons_vals,
        hcva_point=hcva_point,
        transition_point=transition_point,
        )
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
    for society in societies:
        # Separate experiment for each society type
        wandb.init(project="new_examples_runner",
        config={"society": society,
                "timestamp": timestamp,
                }
        )
        for i in range(1, 10):
            # read in data
            #FOR BLUEPEBBLE: data = pd.read_csv('/user/home/ia23938/ValueSystemsAggregation/data/society_data/'+folder+'/'+name+'_'+str(random)+'.csv')
            filename = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/multiple_example_data/'+society+'/'+society+'_'+str(i)+'.csv'
            data = pd.read_csv(filename)

            # Extract values and action judgements for eg. 1
            # need pp, P_1, a_enjoy_camp, a_enjoy_resort, a_budget_camp, a_budget_resort
            personal_data = data[example_data_names]
            principle_data = data[example_principle_names]
            principle_data = principle_data.rename(columns={'egal' : 'rel', 'util' : 'nonrel'})

            experiment_score, decisions, preference_consensus, action_judgement_consensus, transition_point, hcva_point = run_experiment(
                filename=filename, 
                iteration=i, 
                societynum=society)

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