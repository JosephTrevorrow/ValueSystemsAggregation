import pandas as pd

import matplotlib.pyplot as plt

def plot_data(data: pd.DataFrame, title: str):
    """
    This function plots the limit P data (-t True).
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """
    # TODO: THIS IS UNTESTED
    fig, ax = plt.subplots()
    data.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("p")
    ax.set_ylabel("y(p)")
    plt.show()
    return None

if __main__ == "__main__":
    # Example of use
    data = pd.read_csv("p_limits.csv")
    plot_data(data, "Limit P data")