# Import Python Packages
import pandas as pd
import numpy as np
from functions import *
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
T = 50

# Make parameter list to pass to functions.
params = np.array([alpha, beta, delta, n, g, sigma])

# Calculate steady state.
kss_sharp, css_sharp = cal_steady_state(params)
kss_sharp, css_sharp

# Get simulation result.
c_sharp_dev, k_sharp_dev, C_level, K_level, Y_level = simulate_model(
    kss_sharp, css_sharp, params, T)

# Calculate capital-output ratio
K_Y_ratio = K_level/Y_level

################################## Plot ########################################
############ plot time series

# Consumption levels, capital levels, and output levels against time.
time = np.array(list(range(T)))
sns.set_style("white")
sns.set_palette('deep')
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(time, C_level, label="Consumption level")
ax1.plot(time, K_level, label="Capital level")
ax1.plot(time, Y_level, label="Output evel")
ax1.legend(loc='upper left')

ax2.plot(time, K_Y_ratio, color='b', label="Capital output ratio")
ax2.legend(loc='lower right')
plt.xlabel('time')
fig.savefig('figures/time_series_plot')
plt.close()

############ Visualize linearization
# Create a dictionary that maps readable name onto variable names interested.
dict = {'capital deviation': k_sharp_dev, 'consumption deviation': c_sharp_dev,
        'Consumption level': C_level, 'Capital level': K_level, 'Output level': Y_level}

# Define a plot function
def linearize_plot(x_dict_key, y_dict_key, dict):
    '''
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
    '''
    fig, ax = plt.subplots()
    plt.plot(dict[x_dict_key], dict[y_dict_key], marker='x')
    plt.xlabel(x_dict_key)
    plt.ylabel(y_dict_key)
    title = x_dict_key+' vs. '+y_dict_key
    plt.title(title)
    fig.savefig('figures/'+title+'.png')
    plt.close()


# Normalized consumption deviations against normalized capital deviations.
linearize_plot('capital deviation', 'consumption deviation', dict)

# Consumption levels against capital levels.
linearize_plot('Capital level', 'Consumption level', dict)

# Capital levels against output levels.
linearize_plot('Capital level', 'Output level', dict)
