"""
This file will read in 4 random agents from the csv and run through each of the 3 examples in the simulation using their data.
The file will track satisfaction for each agent and store in a results csv file. The simulation will run 4 times, for p=1. p=10, p=transition point, and p=voted_p
"""

import pandas as pd
import solve
from matrices import FormalisationObjects, FormalisationMatrix
    
# Global variables
example_1_personal_data = None
example_1_principle_data = None

#Unfinished
def satisfaction(cons_vals: list, action_cons_vals: list) -> None:
    """
    This function will calculate satisfaction for each agent in the simulation
    """
    satisfaction = {}
    for i in len(example_1_personal_data):
        agent_id = example_1_personal_data.iloc[i]['country']
        # personal data rel - consensus val rel
        satisfaction.update(agent_id : example_1_personal_data.iloc[i][rel] - cons_vals[i][1])

        


def run_experiment() -> None:
    """
    Data is prepared for the sim by using matrices files mimicing original solve.py
    """
    P_list, J_list, w, country_dict = FormalisationObjects(
    filename=None, delimiter=',', df=example_1_personal_data)

    # Solve missing principle values
    #fill_prinicples(personal_vals=args.f, principle_vals=args.pf)
    
    PP_list, PJ_list, Pw, Pcountry_dict = FormalisationObjects(
        filename=None, delimiter=',', df=example_1_principle_data)
    cons_vals = []
    action_cons_vals = []
    for p in [1, 10]:
        # Run simulation for example 1 p = 1
        cons_vals.append(solve.aggregate_values(aggregation_type=True, 
                                        filename="test.csv", 
                                        P_list=P_list, 
                                        J_list=J_list,
                                        w=w,
                                        principle_val=p))
        print("DEBUG: Results for example 1\n", cons_vals)
        action_cons_vals.append(solve.aggregate_values(aggregation_type=False, 
                                        filename="test.csv", 
                                        P_list=P_list, 
                                        J_list=J_list,
                                        w=w,
                                        principle_val=p))
        print("DEBUG: Results for example 1\n", action_cons_vals)

    # Calculate satisfaction (cons_vals, action_cons_vals are in format [p=1, p=10])
    satisfaction(cons_vals=cons_vals, action_cons_vals=action_cons_vals)

if __name__ == '__main__':
    # read in data
    data = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/agent_data.csv')
    
    # Put while loop here
    sample_data = data.sample(n=4)
    
    # Extract values and action judgements for eg. 1
    # need pp, P_1, a_enjoy_camp, a_enjoy_resort, a_budget_camp, a_budget_resort
    example_1_personal_data = sample_data[['agent_id', 'P_1', 'P_1_1', 'a_enjoy_camp', 'a_enjoy_resort', 'a_budget_camp', 'a_budget_resort']]
    example_1_personal_data.rename(columns={'agent_id': 'country', 'P_1': 'rel', 'P_1_1' : 'nonrel', 'a_enjoy_camp' : 'a_adp_rel', 'a_enjoy_resort' : 'a_div_rel', 'a_budget_camp' : 'a_adp_nonrel', 'a_budget_resort' : 'a_div_nonrel'}, inplace=True)
    
    example_1_principle_data = sample_data[['agent_id','pp', 'pp_1']]
    example_1_principle_data.rename(columns={'agent_id': 'country', 'pp': 'rel', 'pp_1' : 'nonrel'}, inplace=True)

    """ Add other examples here:
    example_1_personal_data = sample_data[['agent_id', 'P_1', 'P_1_1', 'a_enjoy_camp', 'a_enjoy_resort', 'a_budget_camp', 'a_budget_resort']]
    example_1_personal_data.rename(columns={'agent_id': 'country', 'P_1': 'rel', 'P_1_1' : 'nonrel', 'a_enjoy_camp' : 'a_adp_rel', 'a_enjoy_resort' : 'a_div_rel', 'a_budget_camp' : 'a_adp_nonrel', 'a_budget_resort' : 'a_div_nonrel'}, inplace=True)
    
    example_1_principle_data = sample_data[['agent_id','pp', 'pp_1']]
    example_1_principle_data.rename(columns={'agent_id': 'country', 'pp': 'rel', 'pp_1' : 'nonrel'}, inplace=True)
    """

    print("DEBUG: Testing with the following sample\n", sample_data)
    run_experiment(sample_data)