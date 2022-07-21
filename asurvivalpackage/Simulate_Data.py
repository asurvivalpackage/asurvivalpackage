import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class Simulate_Data:
    """"
        Class for simulating data from different distributions

        Attributes:
            sample_size: Number of individuals in the sample

        Arguments for simulating_times:
            x: covariates
            coeff: Array of coefficients
            lam: lambda value
            gamma: gamma value - default is 1
            dist: specified distribution
            max_time: time at which administrative censoring occurs - default is none

        """

    def __init__(self, sample_size):
        self.sample_size = sample_size

    # Simulate times method with parameter for dist, covariate values and coefficient value
    def simulate_times_parametric(self, parameters, dist='exponential', max_time=None, censoring=None, cens_parameters=None, covar=None, coeff=None):
        # Multiply covariates by coefficients and sum each row
        # If the covariate array contains only one covariate i.e. one column
        if covar is None:
            covar_total = 0
        else:
            covar_np = np.asarray(covar)
            if len(covar.T) == self.sample_size:
                covar_total = covar_np * coeff
            else:
                covar_total = np.sum(np.multiply(covar_np, coeff), axis=1)

        # Get log(U) and the simulated times according to specified distribution
        U = np.log(np.random.uniform(0, 1, self.sample_size))  # Get random uniform sample of size n between 0 and 1
        if dist == 'exponential':
            lam = parameters[0]
            newtimes = self.exponential_baseline(lam, U, covar_total)  # Transform U into T by the given baseline
            # function
        elif dist == 'weibull':
            lam = parameters[0]
            gamma = parameters[1]
            newtimes = self.weibull_baseline(lam, gamma, U,
                                              covar_total)  # Transform U into T by the given baseline function
        elif dist == 'gompertz':
            lam = parameters[0]
            gamma = parameters[1]
            newtimes = self.gompertz_baseline(lam, gamma, U,
                                               covar_total)  # Transform U into T by the given baseline function
        else:
            raise Exception('Distribution must be exponential, weibull or gompertz.')

        # If no censoring then everyone had the event - event array = 1s
        if max_time is None and censoring is None:
            event_indicator = np.full(self.sample_size, 1)
        elif max_time is not None and censoring is None:
            event_indicator = (newtimes < max_time).astype(int)
            newtimes[newtimes > max_time] = max_time
        elif censoring is not None:
            if max_time is None:
                raise Exception('Must provide a maximum follow up time to include censoring.')
            event_indicator, newtimes = self.censoring(newtimes, max_time, censoring, cens_parameters)

        # Get data frame of data
        data = self.data_frame(newtimes, event_indicator, covar)

        return data

    # Simulate survival times from S(t) predictions
    def simulate_times_model(self, predictions, event_times, max_time, real_data = None, censoring = None, cens_parameters = None):
        # Get number of observations
        rows, columns = predictions.shape
        n = rows

        # Simulate from uniform
        U = np.random.uniform(0,1,n)
        predictions = predictions.T

        # Find S(t) closest to U and corresponding time
        abs_values = np.absolute(predictions-U)
        abs_values = pd.DataFrame(abs_values)
        # Index of the row containing the closest S(t) to U for each individual
        times = abs_values[::-1].idxmin(axis=0).to_numpy() # process rows in reverse order to find last occurence of minimum

        newtimes = []
        i = 0
        # Get the survival time for each individual
        while(i<n):
            index = times[i] # Get the index of the closest S(t) to U for individual_i
            surv_i = predictions[index][i] # Getting the survival probability corresponding to that time

            # Get the event times at which S(ti) > U > S(ti+1)
            if U[i] > surv_i and index != 0:
                if surv_i == predictions[index-1][i]:
                    new_time = event_times[index]
                else:
                    func = interp1d([predictions[index-1][i], surv_i], [event_times[index-1], event_times[index]], kind='linear')
                    new_time = func(U[i])
            elif U[i] > surv_i and index == 0:
                func = interp1d([1, surv_i], [0, event_times[index]])
                new_time = func(U[i])
            elif U[i] < surv_i and index != len(event_times)-1:
                if surv_i == predictions[index+1][i]:
                    new_time = event_times[index]
                else:
                    func = interp1d([surv_i, predictions[index+1][i]], [event_times[index], event_times[index+1]], kind='linear')
                    new_time = func(U[i])
            elif U[i] < surv_i and event_times[index] == np.amax(event_times):
                new_time = max_time
            else:
                new_time = event_times[index]

            newtimes.append(new_time)
            i += 1

        # Censoring
        if censoring is not None:
            event_indicator, newtimes = self.censoring(newtimes, max_time, censoring, cens_parameters)
        else:
            event_indicator = np.less(newtimes, max_time)
            newtimes = np.minimum(newtimes, max_time)

        # Return DataFrame
        if real_data is None:
            data = self.data_frame(newtimes, event_indicator)
        else:
            covar = real_data.drop(columns=['Time', 'Event'])
            cov_names = covar.columns
            data = self.data_frame(newtimes, event_indicator, covar, cov_names)

        return data

    # Method to handle Censoring
    def censoring(self, newtimes, max_time, censoring, cens_parameters):
        n = len(newtimes)

        if cens_parameters is None:
            raise Exception('Must provide parameters for censoring distribution.')
        elif censoring != 'exponential' and censoring != 'weibull':
            raise Exception('Censoring distribution must be exponential or weibull')

        U = np.random.uniform(0,1,n)
        if censoring == 'exponential':
            lam = cens_parameters[0]
            cens_times = -(np.log(U)/lam)
        elif censoring == 'weibull':
            lam = cens_parameters[0]
            gam = cens_parameters[1]
            cens_times = -(np.log(U)/lam)
            cens_times = np.power(cens_times, (np.full(nobs, (1/gam))))

        event_indicator_cens = np.less(newtimes, cens_times)
        event_indicator_admin = np.less(newtimes, max_time)
        event_indicator = np.logical_and(event_indicator_cens, event_indicator_admin).astype(int)
        newtimes = np.minimum(newtimes, cens_times)
        newtimes = np.minimum(newtimes, max_time)

        return event_indicator, newtimes

    def data_frame(self, newtimes, event, covar = None):
        # Append the covariate columns to dataframe
        if covar is None:
            data = {'Time': newtimes,
                    'Event': event}
        else:
            data = covar
            data['Time'] = newtimes
            data['Event'] = event

        data = pd.DataFrame(data)

        return data

    # Simulate a binary variable - random choice of 0 or 1 based on probability p of being 1
    def simulate_binary_covariate(self, values, p):
        covar = np.random.choice(values, self.sample_size, p=[p, 1-p])

        return covar

    # Simulate a continuous variable from distribution dist - default is normal
    def simulate_continuous_covariate(self, parameters, dist='normal'):
        if dist == 'normal':
            mean = parameters[0]
            sd = parameters[1]
            covar = np.random.normal(mean, sd, self.sample_size)
        elif dist == 'uniform':
            minimum = parameters[0]
            maximum = parameters[1]
            covar = np.random.uniform(minimum, maximum, self.sample_size)
        else:
            raise Exception('Distrubtion must be normal or uniform.')

        return covar

    # Distributions for time simulating
    @staticmethod
    def exponential_baseline(lam, U, covar_total):
        t = (-1 * U) / (lam * np.exp(covar_total))
        return t

    @staticmethod
    def weibull_baseline(lam, gamma, U, covar_total):
        powers = np.full(len(U), 1 / gamma)
        times = (-1 * U) / (lam * np.exp(covar_total))
        return np.power(times, powers)

    @staticmethod
    def gompertz_baseline(lam, gamma, U, covar_total):
        # t = 1/gamma * log[1 - (gamma * log(U)/lambda)]
        n = 1 - (gamma * U / (lam * np.exp(covar_total)))
        t = (1 / gamma) * np.log(n)
        return t
