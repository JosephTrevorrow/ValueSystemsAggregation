import pandas as pd
import solve
import csv
import sys
import os
from matrices import FormalisationObjects, FormalisationMatrix


# Global variables
personal_data = None
principle_data = None

# Garbage Collection
#import gc
#gc.set_threshold(0)

def split_into_samples(data, sample_size):
    shuffled_df = data.sample(frac=1).reset_index(drop=True)
    # Split into samples
    samples = [shuffled_df.iloc[i*sample_size:(i+1)*sample_size] for i in range(25)]
    return samples

def satisfaction(decision_made):
    """
    This function will calculate satisfaction for each agent in the simulation
    Satisfaction is calculated as: |(decision made - decision preferred)| for each agent
    An agents ID must be tracked and kept consistent
    """
    satisfaction = {}
    for i in range(len(personal_data)):
        agent_id = personal_data.iloc[i]['country']
        # make agent decision
        agent_decision = solve.make_decision([0, personal_data.iloc[i]['rel'], personal_data.iloc[i]['nonrel'], 0], [0, personal_data.iloc[i]['a_div_rel'], personal_data.iloc[i]['a_div_nonrel'], 0])
        satisfaction.update({agent_id : abs(agent_decision[0] - decision_made[0])})    

    return satisfaction

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
    # I want to have all aggregate data here, and then reason about it afterwards
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
        print(p)
        experiment_cons_vals.append(cons_vals[p])
        experiment_action_cons_vals.append(action_cons_vals[p])
        decision = solve.make_decision(cons_vals[p], action_cons_vals[p])
        experiment_decisions.append(decision)
        score = satisfaction(decision)
        scores.append(score)
    
    return scores, experiment_decisions, experiment_cons_vals, experiment_action_cons_vals, transition_point, hcva_point

    for p in [1, 10, "t", "p"]:
        if p == "t":
            p = solve.transition_point(P_list, J_list, w, country_dict, limit_p_filename)
            transition_point = p
        elif p == "p":
            p = solve.voted_principle(PP_list, PJ_list, Pw, Pcountry_dict, principle_data)
            hcva_point = p
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
        decision = solve.make_decision(cons_vals[-1], action_cons_vals[-1])
        decisions.append(decision)
        score = satisfaction(decision)
        scores.append(score)
    return scores, decisions, cons_vals, action_cons_vals, transition_point, hcva_point

if __name__ == '__main__':
    try:
        filename = sys.argv[1]
    except:
        filename = 'util_dist'

    # read in data
    data = pd.read_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/society_data/'+filename+'.csv')
    
    try:
        path = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_results_v2/'+filename
        os.mkdir(path)
    except OSError as e:
        print("DEBUG: Directory already exists")

    """
    - An experiment will run for 100 days, going out with different agents each time (randomly)
    - We map the satisfaction of each agent over time
    - So every agent will have a satisfaction score for each day
    """
    iterations = 2
    iterator = 0
    experiment_scores = []
    decision_scores = []
    preference_consensuses = []
    action_judgement_consensuses = []
    transition_points = []
    hcva_points = []
    # Situtational data
    example_data_names = [['agent_id', 'P_1', 'P_1_1', 'a_enjoy_camp', 'a_enjoy_resort', 'a_budget_camp', 'a_budget_resort'],
                        ['agent_id', 'P_2', 'P_2_1', 'a_conform_chain', 'a_conform_independent', 'a_stim_chain', 'a_stim_independent'],
                        ['agent_id', 'P_3', 'P_3_1', 'a_enjoy_classic', 'a_enjoy_unknown', 'a_stimulation_classic', 'a_stimulation_unknown'],
    ]
    example_principle_names = [['agent_id', 'pp_1', 'pp_1_1'],
                            ['agent_id', 'pp_2', 'pp_2_1'],
                            ['agent_id', 'pp_3', 'pp_3_1'],
    ]
    while iterator < iterations:
        experiment_scores = []
        decision_scores = []
        preference_consensuses = []
        action_judgement_consensuses = []
        transition_points = []
        hcva_points = []
        print("DEBUG: Iteration: ", iterator)
        # Single sample:
        #sample_data = data.sample(n=4)
        j = 0
        split_samples = split_into_samples(data, sample_size=4)
        for sample_data in split_samples:

            k = 0
            # Loop through each example case
            for (situation, principles) in zip(example_data_names, example_principle_names):
                # Extract values and action judgements for eg. 1
                # need pp, P_1, a_enjoy_camp, a_enjoy_resort, a_budget_camp, a_budget_resort
                sample = sample_data[situation]
                personal_data = sample.rename(columns={'agent_id': 'country', situation[1]: 'rel', situation[2] : 'nonrel', situation[3] : 'a_adp_rel', situation[4] : 'a_div_rel', situation[5] : 'a_adp_nonrel', situation[6] : 'a_div_nonrel'})
                
                sample = sample_data[principles]
                principle_data = sample.rename(columns={'agent_id': 'country', principles[1] : 'rel', principles[2] : 'nonrel'})

                #print("DEBUG: Running experiment for ", personal_data)
                print("DEBUG: Running experiment for ", principle_data)

                experiment_score, decisions, preference_consensus, action_judgement_consensus, transition_point, hcva_point = run_experiment(filename, iterator, j, k)
                experiment_scores.append(experiment_score)
                decision_scores.append(decisions)
                preference_consensuses.append(preference_consensus)
                action_judgement_consensuses.append(action_judgement_consensus)

                transition_points.append(transition_point)
                hcva_points.append(hcva_point)
                k+=1
            # Reset data
            del personal_data
            del principle_data
            j+=1

        solve.shutdown_julia()

        with open(path+filename+'.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            # flatten rows
            for i, sublist in enumerate(experiment_scores):
                for j, dictionary in enumerate(sublist):
                    for key, value in dictionary.items():
                        writer.writerow([i, j, key, value])
        csvfile.close()
        with open(path+filename+'_DECISIONS.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            # flatten rows
            for i, sublist in enumerate(decision_scores):
                for j, decision in enumerate(sublist):
                    writer.writerow([i, j, decision])
        csvfile.close()
        # Store consensus value system
        with open(path+filename+'_CONS_PREFERENCES.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            # flatten rows
            for i, sublist in enumerate(preference_consensuses):
                for j, pref in enumerate(sublist):
                    writer.writerow([i, j, pref])
        csvfile.close()
        with open(path+filename+'_CONS_ACTIONS.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            # flatten rows
            for i, sublist in enumerate(action_judgement_consensuses):
                for j, action in enumerate(sublist):
                    writer.writerow([i, j, action])
        csvfile.close()
        with open(path+filename+'_t_points.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            for point in transition_points:
                writer.writerow([point])
        csvfile.close()
        with open(path+filename+'_hcva_points.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            for point in hcva_points:
                writer.writerow([point])
        csvfile.close()

        # Reset storage
        # Clear memory
        del split_samples
        del experiment_scores
        del decision_scores
        del preference_consensuses
        del action_judgement_consensuses
        del transition_points
        del hcva_points
        #gc.collect()
        # the memory isnt python????? its julia!!!!!!!!!!!!!!
        iterator+=1
        print("DEBUG: Iteration complete")