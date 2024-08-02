"""
This file will generate randomly agent data for 
3 different examples and store in a csv file. in the format:
agent_id, pp (egal, util), P_1 (budget, enjoy), P_2 (conformity, stimulation), P_3 (enjoyment, stimulation),
    a_enjoy(camp), a_enjoy(resort), a_budget(camp), a_budget(resort)
    a_conform(chain), a_conform(independent), a_stim(chain), a_stim(independent)
    a_enjoy(classic), a_enjoy(unknown), a_stimulation(classic), a_stimulation(unknown)
"""

import csv
import random
import pandas as pd

def generate_data(df : pd.DataFrame) -> None:
    for i in range(0, 100):
        agent_id = i
        pp = random.random().__round__(3)
        pp_1 = 1 - pp
        P_1 = random.random().__round__(3)
        P_1_1 = 1- P_1
        P_2 = random.random().__round__(3)
        P_2_1 = 1- P_2
        P_3 = random.random().__round__(3)
        P_3_1 = 1- P_3
        a_enjoy_camp = random.uniform(-1, 1).__round__(3)
        a_enjoy_resort = random.uniform(-1, 1).__round__(3)
        a_budget_camp = random.uniform(-1, 1).__round__(3)
        a_budget_resort = random.uniform(-1, 1).__round__(3)
        a_conform_chain = random.uniform(-1, 1).__round__(3)
        a_conform_independent = random.uniform(-1, 1).__round__(3)
        a_stim_chain = random.uniform(-1, 1).__round__(3)
        a_stim_independent = random.uniform(-1, 1).__round__(3)
        a_enjoy_classic = random.uniform(-1, 1).__round__(3)
        a_enjoy_unknown = random.uniform(-1, 1).__round__(3)
        a_stimulation_classic = random.uniform(-1, 1).__round__(3)
        a_stimulation_unknown = random.uniform(-1, 1).__round__(3)
        
        data = [agent_id, pp, pp_1, P_1, P_1_1, P_2, P_2_1, P_3, P_3_1, a_enjoy_camp, a_enjoy_resort, a_budget_camp, a_budget_resort, a_conform_chain, a_conform_independent, a_stim_chain, a_stim_independent, a_enjoy_classic, a_enjoy_unknown, a_stimulation_classic, a_stimulation_unknown]
        df.loc[len(df)] = data

if __name__ == '__main__':
    df = pd.DataFrame(columns=['agent_id', 'pp', 'pp_1', 'P_1', 'P_1_1', 'P_2', 'P_2_1', 'P_3', 'P_3_1', 'a_enjoy_camp', 'a_enjoy_resort', 'a_budget_camp', 'a_budget_resort', 'a_conform_chain', 'a_conform_independent', 'a_stim_chain', 'a_stim_independent', 'a_enjoy_classic', 'a_enjoy_unknown', 'a_stimulation_classic', 'a_stimulation_unknown'])
    generate_data(df)
    df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/agent_data.csv', index=True)