import pandas as pd
import csv
import numpy as np

def normalise(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

def calculate_individuals_happiness(individual_df):
    # Dict: Agent ID -> Happiness
    happiness_by_agent = {}

    # Compute comparison metrics for each individual (as if were whole country)
    for row in individual_df.iterrows():
        total = row[1]['imptrad'] + row[1]['ipgdtim']
        normalised_religious = row[1]['imptrad'] / total
        normalised_nonreligious = row[1]['ipgdtim'] / total
        individual_df.at[row[0], 'Rel-Nonrel'] = normalised_religious
        individual_df.at[row[0], 'Nonrel-Rel'] = normalised_nonreligious

        # basinc 1-4, 1 is against, 4 is for
        basinc = row[1]['basinc']
        normalised_basinc = (basinc - 1) / 3 * 2 - 1
        individual_df.at[row[0], 'Rel_div_p'] = normalised_religious * normalised_basinc
        individual_df.at[row[0], 'Nonrel_div_p'] = normalised_nonreligious * normalised_basinc
    
    return individual_df


if __name__ == '__main__':
    individual_df = pd.read_spss("/home/ia23938/Documents/GitHub/ValueSystemsAggregation/process_data/ESS/ESS8e02_3-subset.sav", convert_categoricals=False)
    individual_df = individual_df.dropna(subset=['imptrad', 'ipgdtim', 'basinc'])
    # TEMP: Filter individual_df to only include agents in AT
    #individual_df = individual_df[individual_df['cntry'] == 'AT']
    individual_df = individual_df[['idno', 'cntry', 'imptrad', 'ipgdtim', 'basinc']]
    individual_df = calculate_individuals_happiness(individual_df)
    individual_df.to_csv("/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/all_individual_responses.csv", index=False)
