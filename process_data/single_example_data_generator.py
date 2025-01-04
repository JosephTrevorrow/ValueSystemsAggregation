"""
This file will generate randomly agent data for example and store in a csv file. in the format:
agent_id, pp (egal, util), P_1 (budget, enjoy)
    a_enjoy(camp), a_enjoy(resort), a_budget(camp), a_budget(resort)
"""

import csv
import sys
import pandas as pd
import numpy as np

def generate_random_data(df: pd.DataFrame) -> None:
    for i in range(0, 100):
        agent_id = i
        rand_prefs = np.random.uniform(size=2)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = np.random.uniform(low=-1, high=1, size=2)
        data = [agent_id, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df
 
def generate_utilitarian_data(df: pd.DataFrame):
    # Generates agent data that is majority utilitarian
    rng = np.random.default_rng()
    for i in range(0, 100):
        agent_id = i
        principle_pref = rng.normal(loc=0.2, scale=0.25, size=1)
        principle_pref = np.clip(principle_pref, 0, 1)[0]
        principle_prefs = [principle_pref, 1 - principle_pref]
        rand_prefs = rng.normal(loc=0.5, scale=0.25, size=1)
        rand_prefs = np.clip(rand_prefs, 0, 1)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = rng.normal(loc=0, scale=0.5, size=2)
        rand_actions = np.clip(rand_actions, -1, 1)
        data = [agent_id, *principle_prefs, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df

def generate_egalitarian_data(df: pd.DataFrame):
    # Generates agent data that is majority egalitarian
    rng = np.random.default_rng()
    for i in range(0, 100):
        agent_id = i
        principle_pref = rng.normal(loc=0.8, scale=0.25, size=1)
        principle_pref = np.clip(principle_pref, 0, 1)[0]
        principle_prefs = [principle_pref, 1 - principle_pref]
        rand_prefs = rng.normal(loc=0.5, scale=0.25, size=1)
        rand_prefs = np.clip(rand_prefs, 0, 1)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = rng.normal(loc=0, scale=0.5, size=2)
        rand_actions = np.clip(rand_actions, -1, 1)
        data = [agent_id, *principle_prefs, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df

def generate_normal_dist_data(df: pd.DataFrame):
    rng = np.random.default_rng()
    for i in range(0, 100):
        agent_id = i
        rand_prefs = rng.normal(loc=0.5, scale=0.25, size=2)
        rand_prefs = np.clip(rand_prefs, 0, 1)
        rand_prefs = [val for pair in zip(rand_prefs, 1 - rand_prefs) for val in pair]
        rand_actions = rng.normal(loc=0, scale=0.5, size=2)
        rand_actions = np.clip(rand_actions, -1, 1)
        data = [agent_id, *rand_prefs, *rand_actions]
        df.loc[len(df)] = data
    return df

if __name__ == '__main__':
    names_and_functions = {"rand_societies": generate_random_data, "norm_societies": generate_normal_dist_data, "util_societies": generate_utilitarian_data, "egal_societies": generate_egalitarian_data}
    for name, function in names_and_functions.items():
        for i in range(0, 100):
            df = pd.DataFrame(columns=['country','egal','util','rel','nonrel','a_div_rel','a_div_nonrel'])
            function(df)
            df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/multiple_example_data/'+name+'/'+name+'_'+str(i)+'.csv', index=True) 