"""
This file will generate randomly agent data for 
3 different examples and store in a csv file. in the format:
agent_id, pp (egal, util), P_1 (budget, enjoy), P_2 (conformity, stimulation), P_3 (enjoyment, stimulation),
    a_enjoy(camp), a_enjoy(resort), a_budget(camp), a_budget(resort)
    a_conform(chain), a_conform(independent), a_stim(chain), a_stim(independent)
    a_enjoy(classic), a_enjoy(unknown), a_stimulation(classic), a_stimulation(unknown)
"""

import csv
import sys
import pandas as pd
import numpy as np

def generate_random_data(df: pd.DataFrame) -> None:
    for i in range(0, 100):
        agent_id = i
        rand_prefs = np.random.uniform(size=6)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = np.random.uniform(low=-1, high=1, size=12)
        data = [agent_id, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df
 
def generate_utilitarian_data(df: pd.DataFrame):
    # Generates agent data that is majority utilitarian
    rng = np.random.default_rng()
    for i in range(0, 100):
        agent_id = i
        principle_pref = rng.normal(loc=0.2, scale=0.1, size=1)
        principle_pref = np.clip(principle_pref, 0, 1)[0]
        principle_prefs = [principle_pref, 1 - principle_pref]
        rand_prefs = rng.normal(loc=0.5, scale=0.1, size=5)
        rand_prefs = np.clip(rand_prefs, 0, 1)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = rng.normal(loc=0, scale=0.5, size=12)
        rand_actions = np.clip(rand_actions, -1, 1)
        data = [agent_id, *principle_prefs, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df

def generate_egalitarian_data(df: pd.DataFrame):
    # Generates agent data that is majority egalitarian
    rng = np.random.default_rng()
    for i in range(0, 100):
        agent_id = i
        principle_pref = rng.normal(loc=0.8, scale=0.1, size=1)
        principle_pref = np.clip(principle_pref, 0, 1)[0]
        principle_prefs = [principle_pref, 1 - principle_pref]
        rand_prefs = rng.normal(loc=0.5, scale=0.1, size=5)
        rand_prefs = np.clip(rand_prefs, 0, 1)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = rng.normal(loc=0, scale=0.5, size=12)
        rand_actions = np.clip(rand_actions, -1, 1)
        data = [agent_id, *principle_prefs, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df

def generate_normal_dist_data(df: pd.DataFrame):
    rng = np.random.default_rng()
    for i in range(0, 100):
        agent_id = i
        rand_prefs = rng.normal(loc=0.5, scale=0.1, size=6)
        rand_prefs = np.clip(rand_prefs, 0, 1)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = rng.normal(loc=0, scale=0.5, size=12)
        rand_actions = np.clip(rand_actions, -1, 1)
        data = [agent_id, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df

if __name__ == '__main__':
    df = pd.DataFrame(columns=['agent_id', 'pp_1', 'pp_1_1', 'pp_2', 'pp_2_1', 'pp_3', 'pp_3_1', 'P_1', 'P_1_1', 'P_2', 'P_2_1', 'P_3', 'P_3_1', 'a_enjoy_camp', 'a_enjoy_resort', 'a_budget_camp', 'a_budget_resort', 'a_conform_chain', 'a_conform_independent', 'a_stim_chain', 'a_stim_independent', 'a_enjoy_classic', 'a_enjoy_unknown', 'a_stimulation_classic', 'a_stimulation_unknown'])
    #print("DEBUG: All args: ", sys.argv)
    if sys.argv[1] == "-r":
        df = generate_random_data(df)
    elif sys.argv[1] == "-n":
        df = generate_normal_dist_data(df)
    elif sys.argv[1] == "-u":
        df = generate_utilitarian_data(df)
    elif sys.argv[1] == "-e":
        df = generate_egalitarian_data(df)
    else:
        print("No argument found, no data generated")
    df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/agent_data.csv', index=True)