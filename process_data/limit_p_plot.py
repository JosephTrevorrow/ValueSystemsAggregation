import pandas as pd

import matplotlib.pyplot as plt
import scienceplots

def plot_data(data: pd.DataFrame, title: str):
    """
    This function plots the limit P data (-t True).
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """

    plt.figure(figsize=(10, 5))
    plt.style.use("science")
    plt.plot(data["p"], data["Dist_p"], label="$||P^{(1)}_S-P^{(P)}_S||_p$")
    plt.plot(data["p"], data["Dist_inf"], label="$||P^{(\infty)}_S-P^{(P)}_S||_p$")
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"], where=(data["Dist_p"] >= data["Dist_inf"]) & (data["Dist_inf"] >= 0.05), color="blue", alpha=0.3, label="Egalitarian Zone")
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"], where=data["Dist_p"] <= data["Dist_inf"], color="green", alpha=0.3, label="Utilitarian Zone")
    
    #fill dark blue in fully egalitarian zone, $epsilon=0.05$. dist_inf is less than $epsilon$
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"], where=data["Dist_inf"] <= 0.05, color="darkblue", alpha=0.3, label="Fully Egalitarian Zone")
    # Mark the transition point
    transition_point = data.loc[data["Dist_p"] >= data["Dist_inf"], ["p", "Dist_p"]].iloc[0]
    plt.plot(transition_point["p"], transition_point["Dist_p"], "ro", label="Transition Point", markersize=10)
    plt.title(title)
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example of use
    data = pd.read_csv("/home/ia23938/Documents/GitHub/ValueSystemsAggregation/limits.csv")
    plot_data(data, "Value Principle Preference Matrix Aggregation (limit P)")

