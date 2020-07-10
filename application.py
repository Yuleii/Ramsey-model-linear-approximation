# Import Python Packages
import pandas as pd
import numpy as np
from functions import cal_steady_state, simulate_model
import seaborn as sns
import matplotlib.pyplot as plt


# Set parameter values.
alpha = 0.3
beta = 0.95
delta = 0.1
n = 0.01
g = 0.02
sigma = 1.0

# Set the number of time periods to simulate.
T = 100

# Make parameter list to pass to functions.
params = np.array([alpha, beta, delta, n, g, sigma])

# Calculate steady state.
kss_sharp, css_sharp = cal_steady_state(params)

# Get simulation result.
c_sharp_dev, k_sharp_dev, C_level, K_level, Y_level = simulate_model(
    kss_sharp, css_sharp, params, T)

# Calculate capital-output ratio
K_Y_ratio = K_level / Y_level

"""Plot"""

# plot time series.
# Consumption levels, capital levels, and output levels against time.
time = np.array(list(range(T)))

sns.set_style("white")
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 6))

ax1.plot(time, C_level, label="consumption level", color="C0")
ax1.plot(time, K_level, label="capital level", color="C1")
ax1.plot(time, Y_level, label="output level", color="C2")
ax1.legend(loc="upper left")

ax2.plot(time, K_Y_ratio, color="C4", label="capital output ratio")
ax2.legend(loc="lower right")

plt.xlabel("time", fontsize=15)
fig.savefig("figures/time_series_plot")
plt.close()

# Visualize linearization(check whether linearization success)
# Create a dictionary that maps readable name onto variable names interested.
dict = {"capital deviation": k_sharp_dev, "consumption deviation": c_sharp_dev,
        "consumption level": C_level, "capital level": K_level, "output level": Y_level}


# Define a plot function
def linearize_plot(x_dict_key, y_dict_key, dict):
    """Plot the linearized results.

    This function takes two keys from one dictionary and the dictionary per se
    to plot and save the relation betwen the two values corresponding
    to given keys.

    Parameters
    ----------
    x_dict_key: str
        First chosen key of a dictionary.
    x_dict_key: str
        Second chosen key of a dictionary.
    dict: dict
        A dictionary maps readable name onto variable.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("darkgrid")
    sns.scatterplot(x=dict[x_dict_key], y=dict[y_dict_key], ax=ax,
                    alpha=0.7, color="steelblue", s=90, marker="+")
    plt.xlabel(x_dict_key, fontsize=15)
    plt.ylabel(y_dict_key, fontsize=15)
    title = x_dict_key + " vs " + y_dict_key
    plt.title(title, fontsize=18)
    fig.savefig("figures/" + title.replace(" ", "_") + ".png")
    plt.close()


# Normalized consumption deviations against normalized capital deviations.
linearize_plot("capital deviation", "consumption deviation", dict)

# Consumption levels against capital levels.
linearize_plot("capital level", "consumption level", dict)

# Capital levels against output levels.
linearize_plot("capital level", "output level", dict)
