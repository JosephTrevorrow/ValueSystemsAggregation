"""
This file will read in 4 random agents from the csv and run through each of the 3 examples in the simulation using their data.
The file will track satisfaction for each agent and store in a results csv file. The simulation will run 4 times, for p=1. p=10, p=transition point, and p=voted_p
"""

import pandas as pd
import solve
import csv
from matrices import FormalisationObjects, FormalisationMatrix
    
# Global variables
personal_data = None
principle_data = None

def satisfaction(decision_made):
    """
    This function will calculate satisfaction for each agent in the simulation
    Satisfaction is calculated as: |(decision made - decision preferred)| for each agent
    """
    satisfaction = {}
    for i in range(len(personal_data)):
        agent_id = personal_data.iloc[i]['country']
        # make agent decision
        agent_decision = solve.make_decision([0, personal_data.iloc[i]['rel'], personal_data.iloc[i]['nonrel'], 0], [0, personal_data.iloc[i]['a_div_rel'], personal_data.iloc[i]['a_div_nonrel'], 0])
        satisfaction.update({agent_id : abs(agent_decision[0] - decision_made[0])})    

    return satisfaction

def run_experiment() -> None:
    """
    Data is prepared for the sim by using matrices files mimicing original solve.py
    """
    P_list, J_list, w, country_dict = FormalisationObjects(
    filename=None, delimiter=',', df=personal_data)

    # Solve missing principle values
    #fill_prinicples(personal_vals=args.f, principle_vals=args.pf)
    
    print("DEBUG: Principle data\n", principle_data)
    # Note: Could not find JMatrix is printed for preference data (there is no action judgement data)
    PP_list, PJ_list, Pw, Pcountry_dict = FormalisationObjects(
        filename=None, delimiter=',', df=principle_data)
    
    cons_vals = []
    action_cons_vals = []
    
    decisions = []
    scores = {}

    for p in [1, 10, "t", "p"]:
        print("DEBUG: Running with new P = ", p)
        # Run simulation for example 1 p = 1
        if p == "t":
            p = solve.transition_point(P_list, J_list, w, country_dict)
        elif p == "p":
            # get voted principle
            p = solve.voted_principle(PP_list, PJ_list, Pw, Pcountry_dict, principle_data)
        # filename is what you save data as
        cons_vals.append(solve.aggregate_values(aggregation_type=True, 
                                        filename="test.csv", 
                                        P_list=P_list, 
                                        J_list=J_list,
                                        w=w,
                                        principle_val=p))
        action_cons_vals.append(solve.aggregate_values(aggregation_type=False, 
                                        filename="test_action.csv", 
                                        P_list=P_list, 
                                        J_list=J_list,
                                        w=w,
                                        principle_val=p))
        # TODO: Do something with decision.
        decision = solve.make_decision(cons_vals[-1], action_cons_vals[-1])
        decisions.append(decision)
        # Calculate satisfaction (cons_vals, action_cons_vals are in format [p=1, p=10])
        score = satisfaction(decision)
        scores.update(score)
    return scores

if __name__ == '__main__':
    # read in data
    data = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/agent_data.csv')
    
    iterations = 1
    i = 0
    while i < iterations:
        sample_data = data.sample(n=4)
        example_data_names = [['agent_id', 'P_1', 'P_1_1', 'a_enjoy_camp', 'a_enjoy_resort', 'a_budget_camp', 'a_budget_resort'],
                            ['agent_id', 'P_2', 'P_2_1', 'a_conform_chain', 'a_conform_independent', 'a_stim_chain', 'a_stim_independent'],
                            ['agent_id', 'P_3', 'P_3_1', 'a_enjoy_classic', 'a_enjoy_unknown', 'a_stimulation_classic', 'a_stimulation_unknown']
        ]
        example_principle_names = [['agent_id', 'pp_1', 'pp_1_1'],
                                ['agent_id', 'pp_2', 'pp_2_1'],
                                ['agent_id', 'pp_3', 'pp_3_1']
        ]

        experiment_scores = []
        # Loop through each example case
        for (situation, principles) in zip(example_data_names, example_principle_names):
            # Extract values and action judgements for eg. 1
            # need pp, P_1, a_enjoy_camp, a_enjoy_resort, a_budget_camp, a_budget_resort
            sample = sample_data[situation]
            personal_data = sample.rename(columns={'agent_id': 'country', situation[1]: 'rel', situation[2] : 'nonrel', situation[3] : 'a_adp_rel', situation[4] : 'a_div_rel', situation[5] : 'a_adp_nonrel', situation[6] : 'a_div_nonrel'})
            
            # TODO: Change principles from staying the same to changing
            sample = sample_data[principles]
            principle_data = sample.rename(columns={'agent_id': 'country', principles[1] : 'rel', principles[2] : 'nonrel'})

            print("DEBUG: Testing with the following sample\n", personal_data)
            experiment_scores.append(run_experiment())

        # Reset data
        personal_data = None
        principle_data = None

        i+=1
    # Store in a file results
    headers = experiment_scores[0].keys()
    print("DEBUG: Headers\n", headers)
    with open('test4.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = headers)
        writer.writeheader()
        writer.writerows(experiment_scores)
