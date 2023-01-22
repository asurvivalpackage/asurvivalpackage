import numpy as np
import math
import scipy.optimize
import pandas as pd

# Handle RunTimeWarnings as exceptions
import warnings

warnings.filterwarnings('error')


class Flexible_Parametric_Models:
    """
        Class for fitting restricted cubic splines

        Attributes: data: Data that the restricted cubic splines are fit to - dataframe with Time, Event indicator
        and then covariates

            Optional from user:
                df: Degrees of freedom chosen by user, default is 1
                knots: Knots specified by user - must be ARRAY INCLUDING BOUNDARY KNOTS

            n_spline_terms: number of spline terms in basis function matrix i.e. (df + 1)
            x: Times
            events: 0 or 1 for if censored or event respectively
            covars: covariate list - this needs to be updated so multiple covariates
            num_covars: Number of covariates in model
            est_of_params: default is None, updated once fitting method has been used
    """

    def __init__(self):
        self.data = None
        self.df = None
        self.knots = None
        self.n_spline_terms = None
        self.x = None
        self.events = None
        self.covars = None
        self.covars_names = None
        self.num_covars = 0
        self.est_of_params = None

    def set_self(self, data, df, knots, tolerance):
        #data['Time'] = np.log(data['Time'])
        self.data = data
        self.df = df
        self.knots = knots
        self.n_spline_terms = self.df + 1

        # Get the times, events and covariates from the dataframe
        self.x = np.log(self.data['Time'])
        self.events = self.data['Event']
        self.covars = data.drop(columns={'Time', 'Event'})
        self.covars_names = self.covars.columns

        self.covars = self.covars.to_numpy()  # Change covariates into matrix format

        # Get the number of covariates
        if self.covars.ndim == 1:
            self.num_covars = 1
        else:
            self.num_covars = np.shape(self.covars)[1]

        # Set initial estimate of parameters to empty array
        self.est_of_params = None

    # Main restricted cubic spline method to fit the FP to the dataset given df
    def fit_FP(self, data, df=3, knots=None, tolerance=1e-14, verbose=False):
        # Set all attibutes
        self.set_self(data, df=df, knots=knots, tolerance=tolerance)

        # Get the knot locations:
        self.handle_knots()

        # Get spline matrix
        spline_matrix = self.get_spline_matrix()

        # Get the design matrix
        design_matrix = np.c_[spline_matrix, self.covars]

        # Get the derivative matrix
        deriv_matrix = self.get_deriv_matrix()

        # Minimise the negative log-likelihood to get parameter values
        n_parameters = self.n_spline_terms + self.num_covars  # Number of parameters = spline parameters + covariate
        parameters = np.full(n_parameters, 0.0)  # log scale i.e. log(param) = 0
        parameters[0] = math.log(1)
        parameters[1] = 0.1

        scipy_output = scipy.optimize.minimize(self.log_likelihood, parameters, args=(design_matrix, deriv_matrix),
                                               method='Nelder-Mead', tol=tolerance, options={'maxiter': 10000})

        est_params = scipy_output.get('x')

        # If user specifies to print details of optimisation, print them
        if verbose is True:
            print(scipy_output)

        # Set estimates attribute of FP object to the estimated parameters - use in predicting at time points
        self.est_of_params = est_params

        parameters = self.parameters_data_frame(np.exp(est_params), self.covars_names, self.n_spline_terms)

        #return self.est_of_params
        return parameters

    def handle_knots(self):
        # If the user hasn't specified knots, calculate them using the df
        if self.knots is None:
            self.knots = self.get_knots()
        # If user has specified knots, ensure they match the df, if not use default knot locations
        elif len(self.knots) != (self.df + 1):
            raise Exception('Number of knots given does not match the degrees of freedom specified.')
        else:
            self.knots = np.log(self.knots)

    # Method to get the knot locations
    def get_knots(self):
        # Get the sorted times of only the people who were not censored
        event_data = self.data.loc[self.data['Event'] == 1]
        event_times = event_data['Time']
        sorted_times = np.sort(event_times)
        max_time = np.amax(event_times)
        min_time = np.amin(event_times)

        knots = [min_time]
        internal_knots = []

        # Get the percentiles of the internal knots and the internal knot values
        if self.df != 1:
            percentiles = []
            knot_space = 100 / self.df
            knot = knot_space
            percentiles.append(knot_space)
            i = 1
            while i < self.df - 1:
                knot = knot + knot_space
                percentiles.append(knot)
                i += 1
            # Get the internal knot values
            internal_knots = np.percentile(sorted_times, percentiles)

        knots = np.concatenate([knots, internal_knots])
        # Update the class attribute
        self.knots = np.append(knots, max_time)

        return self.knots

    # Method to get the internal knots, kmin and kmax and lambda value
    def get_min_max_knots(self):
        kmin = self.knots[0]  # Minimum knot value
        kmax = self.knots[-1]  # Maximum knot value

        # Get the internal knots and the lambda values
        internal_knots = []
        lam_val = 0
        if self.n_spline_terms >= 1:
            internal_knots = self.knots[1:self.df]  # Array of knot 2 to knot max - only internal knots
            lam_val = (kmax - internal_knots) / (kmax - kmin)

        return kmin, kmax, internal_knots, lam_val

    # Method to get the spline matrix
    def get_spline_matrix(self, times=None):
        # If times is specified by user/predict method use that x, if not use the object attribute x
        if times is None:
            x_method = self.x.copy()  # Use a copy of the self.x array so attribute isn't changed
        else:
            x_method = times

        # Number of observations in times given to method - this changes when predict method
        n_obs = len(x_method)

        kmin, kmax, internal_knots, lam_val = self.get_min_max_knots()

        spline_matrix = np.empty((n_obs, self.n_spline_terms))  # Empty matrix for spline values

        # Set the first column in the matrix to 1 - intercept value will go here
        spline_matrix[:, 0] = 1

        # Set the second column in the matrix to be the time value (ln(x)) - Z1
        spline_matrix[:, 1] = x_method

        m = len(internal_knots)
        # Calculate the values in the matrix for Z2,....,Zn
        if self.df > 1:
            for j in range(0, m):
                # j = the column in the spline matrix so we need j+2 as we have the intercept and V1 value (ln(t))
                spline_matrix[:, j + 2] = np.power((x_method - internal_knots[j]), 3) * (x_method > internal_knots[j]) \
                                          - lam_val[j] * np.power((x_method - kmin), 3) * (x_method > kmin) - \
                                          (1 - lam_val[j]) * np.power((x_method - kmax), 3) * (x_method > kmax)

        return spline_matrix

    # Get the derivative matrix of the spline functions
    def get_deriv_matrix(self, times=None):
        # If x is specified by user/predict method use that x, if not use the object attribute x
        if times is None:
            x_method = self.x.copy()  # Use a copy of the self.x array so attribute isn't changed
        else:
            x_method = times

        # Number of observations in times given to method - this changes when predict method
        n_obs = len(x_method)

        deriv_matrix = np.empty((n_obs, self.df))  # Empty matrix for the derivatives

        kmin, kmax, internal_knots, lam_val = self.get_min_max_knots()

        # Set first column to be 1 (first deriv term is just the gamma_1 value)
        deriv_matrix[:, 0] = 1

        # Get the derivative values for Z2,....,Zn
        m = len(internal_knots)
        if self.df > 1:
            for j in range(0, m):
                # j = column in the derivative matrix so need j+1 as we have V1 for the ln(t) derivative
                deriv_matrix[:, j + 1] = 3 * np.power((x_method - internal_knots[j]), 2) * (
                        x_method > internal_knots[j]) - \
                                         3 * lam_val[j] * np.power((x_method - kmin), 2) * (x_method > kmin) - \
                                         3 * (1 - lam_val[j]) * np.power((x_method - kmax), 2) * (x_method > kmax)

        return deriv_matrix

    # Get the log-likelihood for a set of parameters
    def log_likelihood(self, parameters, design_matrix, deriv_matrix):
        # Parameters are the alpha values
        param_no_covar = len(parameters) - self.num_covars  # Parameters not including covariate
        derivative_params = parameters[1:param_no_covar]  # a_1 as in loglikelihood we have log(gamma) as derivative

        spline_parameters = parameters[:]

        # Log time
        log_t = design_matrix[:, 1]

        # Multiply the design matrix by the spline parameters
        design_matrix = np.multiply(design_matrix, spline_parameters)

        # Multiply the derivative matrix by the derivative parameters - log(gamma)
        deriv_matrix = np.multiply(deriv_matrix, derivative_params)
        sum_deriv = np.sum(deriv_matrix, axis=1)  # Sum the columns along for each row - get a vector here

        # Calculate the log_likelihood
        hazard = (-1 * log_t) + np.log(sum_deriv) + np.sum(design_matrix, axis=1)
        hazard = np.multiply(self.events, hazard)
        survival = np.exp(np.sum(design_matrix, axis=1))
        log_likelihood = -1 * np.sum(hazard - survival)

        return log_likelihood

    # Predict survival, hazard and cumulative hazard function at a time point using estimates for splines
    # baseline=True - all covariates are set to 0
    def predict_FP(self, predict_times, baseline=False, covariates=None, scale='log'):
        # Sort out the covariates
        if covariates is not None:
            covariates = np.full((len(predict_times), self.num_covars), covariates)  # Make into matrix of covariates
        elif covariates is None and baseline is True:
            covariates = np.zeros((len(predict_times), self.num_covars))  # Matrix of 0s
        elif baseline is False and covariates is None and self.num_covars != 0:
            raise Exception(
                'Covariates cannot be None if Baseline=False - there are covariates included in this model.')

        # Sort out the times
        # Find t in times that are == 0, survival here is 1 and log(cumulativeH(t)) = 0
        predict_times = np.array(predict_times).astype(np.float32)
        indexes = np.where(predict_times == 0)[0]

        # If log change any 0s to 0.0001 to avoid error (default this survival to 1 anyway)
        if scale == 'log':
            for i in indexes:
                predict_times[i] = 0.0001
            predict_times = np.log(predict_times)

        survival, hazard, log_cum_hazard = self.calculate_predictions(predict_times, covariates)

        # Set time 0 function values to 1 and -inf for S(t) and log H(t)
        for i in indexes:
            survival[i] = 1
            log_cum_hazard[i] = - math.inf
            predict_times[i] = - math.inf

        predictions = self.pred_data_frame(predict_times, baseline, covariates, survival, hazard, log_cum_hazard)

        return predictions

    # Method to calculate the prediction values
    def calculate_predictions(self, times, covariates):

        # Get the spline matrix for the time_point
        spline_matrix = self.get_spline_matrix(times)

        # Get the design matrix - covariates is numpy matrix
        if covariates is None and self.num_covars == 0:  # If there are no covars and there are none in the model
            design_matrix = spline_matrix
        else:
            design_matrix = np.c_[spline_matrix, covariates]

        # Multiply the design matrix by the estimated parameters - n
        spline_values = np.multiply(design_matrix, self.est_of_params)

        n = np.sum(spline_values, axis=1)  # Add up spline values by row

        # Get the log_cumulative hazard at each time_point - if t = 0, then log_cum_h = 0
        log_cum_hazard = n

        # Get the survival function at each time_point
        survival = np.exp(-1 * np.exp(n))

        # Get the derivative matrix and deriv parameters
        deriv_matrix = self.get_deriv_matrix(times)
        param_no_covar = len(self.est_of_params) - self.num_covars
        parameters = self.est_of_params.copy()  # Copy of parameters so object attribute doesn't change
        derivative_params = parameters[1:param_no_covar]

        # Multiply derivative with derivative parameter estimates
        deriv_values = np.multiply(deriv_matrix, derivative_params)

        # Get the hazard function at each time_point
        deriv_values = np.sum(deriv_values, axis=1)  # Sum the derivative across each row
        hazard = 1/np.exp(times) * np.multiply(deriv_values, np.exp(n))  # deriv * exp(design matrix *params)

        return survival, hazard, log_cum_hazard

    @staticmethod
    def parameters_data_frame(parameters, covars_names, n_spline_terms):
        parameters_df = []
        for i in range(0, len(parameters)):
            if(i<1):
                name = 'Intercept'
            elif(0<i<=(n_spline_terms-1)):
                name = 'Spline Term S' + str(i)
            elif(i > (n_spline_terms-1)):
                j = i - n_spline_terms
                name = covars_names[j]
            parameters_df.append({'Parameter': name,
                                  'Estimate': parameters[i]})

        parameters_df = pd.DataFrame(parameters_df)
        return parameters_df

    @staticmethod
    def pred_data_frame(times, baseline, covariates, survival, hazard, log_cum_hazard):
        data = {'Time': np.exp(times)}
        predictions = pd.DataFrame(data)

        if baseline is False:
            # Append the covariate columns to dataframe
            # If the covariate array contains only one covariate i.e. one column
            if len(covariates.T) == len(times):
                n = len(predictions.columns)
                predictions.insert(loc=n, column='Covariate', value=covariates)
            else:
                for i in range(0, len(covariates.T)):
                    var = covariates[:, i]
                    n = len(predictions.columns)
                    name = 'Covariate_' + str(n - 1)
                    predictions.insert(loc=n, column=name, value=var)

        predictions['S(t)'] = survival
        predictions['h(t)'] = hazard
        predictions['log H(t)'] = log_cum_hazard

        return predictions
