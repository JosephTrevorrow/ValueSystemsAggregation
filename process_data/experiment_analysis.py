"""
This file creates graphs tracking satisfaction over time for each agent in the simulation
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

def plot_agent_satisfaction(data: pd.DataFrame, title: str):
    """
    This function plots the satisfaction data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """


def plot_average_satisfaction(data: pd.DataFrame, title: str):
    """
    This function plots the average satisfaction for each test in the data.
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """

def unpack_data(filename: str):
    unpacked_data = defaultdict(lambda: defaultdict(list))
    
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            agent_id = row[0]
            iteration = row[1]
            satisfaction = row[2]

            unpacked_data[agent_id][iteration].append(satisfaction)
        final_data = [[dict(unpacked_data)]]
    return final_data


if __name__ == "__main__":
    # Example of use
    data = unpack_data("/home/ia23938/Documents/GitHub/ValueSystemsAggregation/results_v1/egalsoc_10iteration_3diffagent.csv")
    plot_average_satisfaction(data, "Agent Satisfaction Over Time")