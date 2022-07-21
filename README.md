# asurvivalpackage

Survival analysis package in Python. Includes: Kaplan-Meier, Parametric models, Flexible Parametric models, and simulating survival data.
This package was created to implement Flexible Parametric modelling for time-to-event data in Python as well as a novel approach to simulating survival data from risk prediction models.


## Setup

You need `python>=3.6.4` to use this package.

The package depends on the `pandas`, `numpy` and `scipy` libraries. Install with `pip` e.g:

`pip install pandas`.


## Methods

Includes: Kaplan-Meier, Parametric models, Flexible Parametric models, and simulating survival data.

|Class|Method|Description|
|----|----|----|
|`Kaplan_Meier`|`fit_kaplan_meier(self, data, CI=True)`|Calculates Kaplan-Meier estimates of survival for a dataset|
| |`fit_kaplan_meier_two_groups(self, data, covariate, CI=True)`|Calculates Kaplan-Meier estimates of survival for two groups in a dataset|
|`Parametric_Models`|`fit_model(self, data, model='exponential', init_parameters=None, max_iter=1000, convergence=0.001)`|Estimates a specified parametric model.|
||`predict(self, predict_times)`|Predicts survival, hazard and log cumulative hazard function estimates from the estimated parametric model.|
|`Flexible_Parametric_Models`|`fit_FP(self, data, df=1, knots=None, tolerance=1e-6, verbose=False)`|Estimates a flexible parametric model.|
||`predict_FP(self, predict_times, scale=None, baseline=False, covariates=None)`|Predicts the survival, hazard and log cumulative hazard functions at specified time points given the estimated flexible parametric model.|
|`Simulate_Data`|`simulate_binary_var(self, values, p)`|Simulates an array of binary covariates.|
||`simulate_cont_var(self, parameters, dist='normal')`|Simulates an array of continuous covariates.|
||`simulate_times_parametric(self, lam, gamma=1, dist='exponential', max_time=None, covar=None, cov_names=None, coeff=None, censoring=None, cens_parameters=None)`|Simulates survival data from a specified parametric distribution.|
||`simulate_times_model(self, predictions, event_times, max_time, real_data = None, censoring = None, cens_parameters = None)`|Simulates survival data using the given survival predictions from a risk prediction model.|
