# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:50:17 2023

@author: hrs18
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:50:17 2023

@author: hrs18
"""

# Example for all methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Kaplan_Meier import Kaplan_Meier
from Parametric_Models import Parametric_Models
from Flexible_Parametric_Models import Flexible_Parametric_Models
from Simulate_Data import Simulate_Data

# Read in the RGBC data
data = pd.read_csv('~/asurvivalpackage/RGBC_data.csv', index_col=False)
data = data.rename(columns={'t': 'Time', 'd': 'Event'})
data = data.drop(columns=['Unnamed: 0', 'pid'])
pd.set_option('display.max_columns', None)
print(data.head())


######## Kaplan_Meier Example ########
# KM one group
km = Kaplan_Meier()
km_estimates = km.fit_kaplan_meier(data, CI=True)
print(km_estimates.head())


# Plot with matplotlib
fig, ax = plt.subplots()
ax.step(km_estimates['Time'], km_estimates['S(t)'], linewidth=2.5, color='#77AADD')
ax.fill_between(km_estimates['Time'], km_estimates['Lower CI'], km_estimates['Upper CI'], color='#77AADD', alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('S(t)')
ax.grid()
ax.set_title('Kaplan-Meier Curve')
ax.set_xlim([0, 15])
ax.set_ylim([0, 1])
fig.set_size_inches(8.3,5.8)
plt.savefig('KM_Curves.pdf')
plt.show()

# KM Two groups
km_group1, km_group2 = km.fit_kaplan_meier_two_groups(data, covariate='chemo', CI=True)
print(km_group1.head())
print(km_group2.head())

# Plot two km curves with matplotlib
fig, ax = plt.subplots()
ax.step(km_group1['Time'], km_group1['S(t)'], linewidth=2.5, color='#77AADD', label='Chemo=0')
ax.step(km_group2['Time'], km_group2['S(t)'], linewidth=2.5, color='#44BB99', label='Chemo=1')
ax.fill_between(km_group1['Time'], km_group1['Lower CI'], km_group1['Upper CI'], color='#77AADD', alpha=0.3)
ax.fill_between(km_group2['Time'], km_group2['Lower CI'], km_group2['Upper CI'], color='#44BB99', alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('S(t)')
ax.grid()
ax.set_title('Kaplan-Meier Curve')
ax.set_xlim([0, 15])
ax.set_ylim([0, 1])
fig.set_size_inches(8.3,5.8)
plt.legend()
plt.show()



######## Parametric Models Example ########
pm = Parametric_Models()
weibull_estimates = pm.fit_model(data, model='weibull', init_parameters=[1,1])
print(weibull_estimates)

predictions = pm.predict(predict_times=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(predictions)

# Plot with matplotlib - km plot overlayed
fig, ax = plt.subplots()
ax.plot(predictions['Time'].to_numpy(), predictions['S(t)'].to_numpy(), linewidth=2.5, color='#44BB99', label='Weibull Model')
ax.step(km_estimates['Time']to_numpy(), km_estimates['S(t)'].to_numpy(), linewidth=2.5, color='#77AADD', label = 'Kaplan-Meier')
ax.set_xlabel('Time')
ax.set_ylabel('S(t)')
ax.grid()
ax.set_title('Weibull Model Predicted S(t)')
ax.set_xlim([0, 15])
ax.set_ylim([0, 1])
fig.set_size_inches(8.3,5.8)
plt.legend()
plt.show()

######## Flexible Parameteric Models Example ########
fp_data = data[['Time', 'Event', 'age', 'nodes', 'pr', 'er']]

fp = Flexible_Parametric_Models()
fp_estimates = fp.fit_FP(fp_data, df=3, verbose=False)
fp_estimates.to_csv('~/asurvivalpackage/Example/FP_estimates.txt', sep='\t', index=False)
print(fp_estimates)

predict_times = np.linspace(0, 15, num=50)
predictions = fp.predict_FP(predict_times=predict_times, baseline=True, scale='log')
print(predictions.head())

predictions.to_csv('~/asurvivalpackage/Example/FP_Predictions.txt', sep='\t', index=False)

# Plot S(t) with plotly
fig, ax = plt.subplots()
ax.plot(predictions['Time'].to_numpy(), predictions['S(t)'].to_numpy(), linewidth=2.5, color='#44BB99', label='Flexible Parametric Model')
ax.step(km_estimates['Time'].to_numpy(), km_estimates['S(t)'].to_numpy(), linewidth=2.5, color='#77AADD', label = 'Kaplan-Meier')
ax.set_xlabel('Time')
ax.set_ylabel('S(t)')
ax.grid()
ax.set_title('Flexible Parametric model Predicted S(t)')
ax.set_xlim([0, 15])
ax.set_ylim([0, 1])
fig.set_size_inches(8.3,5.8)
plt.legend()
plt.show()

# Plot h(t) with plotly
fig, ax = plt.subplots()
ax.plot(predictions['Time'].to_numpy(), predictions['h(t)'].to_numpy(), linewidth=2.5, color='#77AADD')
ax.set_xlabel('Time')
ax.set_ylabel('h(t)')
ax.grid()
ax.set_title('Flexible Parametric model Predicted h(t)')
ax.set_xlim([0, 15])
ax.set_ylim([0, 0.05])
fig.set_size_inches(8.3,5.8)
plt.legend()
plt.show()

predict_times = [10]
covars = [65, 1, 35, 12]
predictions = fp.predict_FP(predict_times=predict_times, baseline=False, covariates = covars, scale='log')
print(predictions.head())

predictions.to_csv('~/asurvivalpackage/Example/FP_Predictions_Covars.txt', sep='\t', index=False)


######## Simulate Data ########
# Parametric
sim = Simulate_Data(sample_size=10)

binary = sim.simulate_binary_covariate(values=[0,1], p=0.5)
print(binary)

continuous = sim.simulate_continuous_covariate(parameters=[60,10], dist='normal')
print(continuous)

covariates = pd.DataFrame({'Treatment': binary,
                          'age': continuous})

weibull_data = sim.simulate_times_parametric(parameters=[0.2,2], dist='weibull',
                    max_time=10, censoring='exponential', cens_parameters=[0.001],
                    covar=covariates, coeff=[-0.5, 0.1])
print(weibull_data)
weibull_data.to_csv('~/asurvivalpackage/Example/Weibull_Data.txt', sep='\t', index=False)


# From risk prediction model
events = data[data['Event']==1]
event_times = np.unique(events['Time'])

predictions = pd.read_csv('~/asurvivalpackage/Example/predictions_CoxCC.csv', index_col=False, header=None)
predictions = np.array(predictions)
sim_model = Simulate_Data(sample_size = 2982)

weibull_sim_data = sim_model.simulate_times_model(predictions, event_times,
    max_time=15, real_data = data, censoring = 'exponential', cens_parameters = [0.001])

weibull_sim_data.to_csv('~/asurvivalpackage/Example/CoxCC_Data.txt', sep='\t', index=False)
