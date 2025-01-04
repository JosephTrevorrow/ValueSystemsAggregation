"""
This file is used to run experiments and calculate performance metrics on the data in realtime. The code is taken from other jupyter notebooks
in the repository, replicating the same performance metric logging in the `/process_data/single_example_data_analysis` notebooks
1. Data is plucked from the `hpc_runs_society_data` folder, and data id is logged
2. The dataset is aggregated on, and the HCVA point and transition point are calculated, and logged
3. The performance of each strategy as described in the `single_example_data_analysis_notebooks` folder is calculated and logged for each strategy
"""

"""
OLD EXAMPLES RUNNER
- experiments are ran for a number of days over a number of different contexts
- each experiment is logged locally and on wandb (local logging is currently commented out)
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
    for p in [1, 10, transition_point, hcva_point]:
        # Convert p value to point in list (list is 0.1 increments)
        p = int((p-1)*10)
        # TODO: Place all data analysis here
        decision = solve.make_decision(cons_vals[p], action_cons_vals[p])

    
    return scores, experiment_decisions, experiment_cons_vals, experiment_action_cons_vals, transition_point, hcva_point

########
# Main #
########
if __name__ == '__main__':
    # Columns to be used in aggreagation and analysis
    example_data_names = ['agent_id', 'P_1', 'P_1_1', 'a_enjoy_camp', 'a_enjoy_resort', 'a_budget_camp', 'a_budget_resort']
    example_principle_names = ['agent_id', 'pp_1', 'pp_1_1']

    # Types of societies to be looked at. Each dataset is created to have principles with different overall trends
    society_names = ['util_society', 
                     'egal_society', 
                     'norm_society', 
                     'rand_society']
    societies = {'util_society':'util_societies', 
                 'egal_society':'egal_societies', 
                 'norm_society':'norm_societies', 
                 'rand_society':'rand_societies'}

    timestamp = date.today().strftime('%Y-%m-%d') 

    # Sanity check to stop wandb to stop overrunning
    if wandb.run is not None:
        wandb.finish()
    
    # Iterate over all different society types, to get data for each
    for name, folder in societies.items():
        # Separate experiment for each society type
        wandb.init(project="valuesystemsaggregation",
        config={"society": name,
                "timestamp": timestamp,
                }
        )
        for i in range(1, 50):
            # read in data
            #FOR BLUEPEBBLE: data = pd.read_csv('/user/home/ia23938/ValueSystemsAggregation/data/society_data/'+folder+'/'+name+'_'+str(random)+'.csv')
            data = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/society_data_reformatted/'+folder+'/'+name+'_'+str(random)+'.csv')

            # Extract values and action judgements for eg. 1
            # need pp, P_1, a_enjoy_camp, a_enjoy_resort, a_budget_camp, a_budget_resort
            personal_data = sample.rename(columns={'agent_id': 'country', situation[1]: 'rel', situation[2] : 'nonrel', situation[3] : 'a_adp_rel', situation[4] : 'a_div_rel', situation[5] : 'a_adp_nonrel', situation[6] : 'a_div_nonrel'})
            
            principle_data = sample.rename(columns={'agent_id': 'country', principles[1] : 'rel', principles[2] : 'nonrel'})

            experiment_score, decisions, preference_consensus, action_judgement_consensus, transition_point, hcva_point = run_experiment(name, iterator, context_num, sample_num)

            # log metrics to wandb
            wandb.log({"iteration": iterator, 
                        hcva_score,
                        transition_score,
                        
                        })
            # Clear memory
            try:
                solve.shutdown_julia()
                del personal_data, principle_data
                #gc.collect()
            except Exception as e:
                print("DEBUG: Error in memory clearing, exception:", e)
    
    print("DEBUG: Experiment complete")