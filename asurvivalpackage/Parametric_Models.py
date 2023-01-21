import numpy as np
import math
import pandas as pd


class Parametric_Models:
    """
    Class for fitting an exponential, weibull and gompertz model to a dataset.

    Attributes:

    """

    def __init__(self):
        # Initial method to init the object - set all attributes to null
        self.model = None
        self.time = None
        self.event = None
        self.init_parameters = None
        self.max_iter = 0
        self.convergence = 0

        self.parameters = []
        self.parameters_exp = []

    # Method to fit a model to some dataset
    def fit_model(self, data, model='exponential', init_parameters=None, max_iter=1000, convergence=1e-14):
        self.model = model
        self.time = data['Time']
        self.event = data['Event']
        self.max_iter = max_iter
        self.convergence = convergence

        if init_parameters is None:
            if model == 'exponential':
                init_parameters = [1]
            else:
                init_parameters = [1,1]

        # Check parameters
        if 0 in init_parameters:
            raise Exception('Initial parameter values cannot be 0')
        elif model == 'exponential' and len(init_parameters) > 1:
            raise Exception('Only one parameter in an exponential model, only provide one intial parameter value')
        elif (model == 'weibull' or model == 'gompertz') and len(init_parameters) != 2:
            raise Exception('Two initial parameter values required for ' + model + ' model')

        # Log the initial parameters for estimation
        if 0 in init_parameters:
            index = np.where(init_parameters == 0)[0]
            init_parameters[index] = 1  # Change any initial parameters that are 0 to 1 for logging

        self.init_parameters = np.log(init_parameters)

        if model == 'exponential':
            self.parameters = self.estimate_single_parameter()
        elif model == 'weibull':
            jacobian = [self.f_weibull_lambda, self.f_weibull_gamma]
            hessian = [[self.ff1_weibull_lambda, self.ff_weibull_gl], [self.ff_weibull_gl, self.ff2_weibull_gamma]]
            self.parameters = self.estimate_multiparameters(jacobian, hessian)
        elif model == 'gompertz':
            jacobian = [self.f_gompertz_lambda, self.f_gompertz_gamma]
            hessian = [[self.ff1_gompertz_lambda, self.ff_gompertz_gl], [self.ff_gompertz_gl, self.ff2_gompertz_gamma]]
            self.parameters = self.estimate_multiparameters(jacobian, hessian)
        else:
            raise Exception('Distribution must be Exponential, Weibull or Gompertz.')

        # Exponentiate parameter estimates
        self.parameters_exp = np.exp(self.parameters)

        return self.parameters_exp

    # Method to estimate a single parameter - takes initial log(theta) value
    def estimate_single_parameter(self):
        iteration = 0
        theta = self.init_parameters
        while iteration < self.max_iter:
            iteration += 1
            theta_1 = theta + (self.f_exponential(theta) / (-1 * (self.ff_exponential(theta))))
            if abs(theta_1 - theta) <= self.convergence:
                break
            theta = theta_1

        return theta

    # Estimate multi-parameter values
    def estimate_multiparameters(self, jacobian, hessian):

        iteration = 0
        theta = self.init_parameters
        while iteration < self.max_iter:
            iteration += 1
            jacob_values = self.calculate_jacob(jacobian, theta)  # Calculate first derivatives values
            hess_values = self.calculate_hessian(hessian,
                                                 theta)  # Calculate values for the hessian i.e. second derivatives
            hess_inverse = np.linalg.pinv(hess_values)
            func = np.matmul(hess_inverse, jacob_values)
            theta_1 = theta - func

            d = abs(theta_1 - theta)
            if np.sum(d) < self.convergence:
                break

            theta = theta_1

        return theta

    # Methods to predict survival, hazard and cumulative hazard?
    def predict(self, predict_times):
        # Get index where predict_times = 0
        # Find t in times that are == 0, survival here is 1 and log(cumulativeH(t)) = 0
        predict_times = np.array(predict_times).astype(np.float32)
        indexes = np.where(predict_times == 0)[0]

        # Avoid divide by zero error
        for i in indexes:
            predict_times[i] = 0.1  # To avoid divide by zero error - manually set at time 0

        if self.model == 'exponential':
            lam = self.parameters_exp[0]
            survival = np.exp(-1 * lam * predict_times)
            h = np.full(len(predict_times), lam)
            H = lam * predict_times
        elif self.model == 'weibull':
            lam = self.parameters_exp[0]
            gam = self.parameters_exp[1]
            gam_powers = np.full(len(predict_times), gam)
            survival = np.exp(-1 * lam * np.power(predict_times, gam_powers))
            h = lam * gam * np.power(predict_times, (gam_powers - 1))
            H = lam * np.power(predict_times, gam_powers)
        elif self.model == 'gompertz':
            lam = self.parameters_exp[0]
            gam = self.parameters_exp[1]
            survival = np.exp(-1 * lam * (1 / gam) * (np.exp(gam * predict_times) - 1))
            h = lam * np.exp(gam * predict_times)
            H = lam * (1 / gam) * (np.exp(gam * predict_times) - 1)
        else:
            raise Exception('No model has been fitted.')

        # Where predict_times = 0, change surv = 1 and H(t) = 0
        for i in indexes:
            survival[i] = 1
            H[i] = 0
            predict_times[0] = 0  # Change back to 0 for printing

        predictions = self.pred_data_frame(predict_times, survival, H, h)

        return predictions

    @staticmethod
    def pred_data_frame(predict_times, survival, H, h):
        data = {'Time': predict_times,
                'S(t)': survival,
                'log H(t)': H,
                'h(t)': h}

        predictions = pd.DataFrame(data)

        return predictions

    # Calculate values for the jacobian i.e. first derivatives
    @staticmethod
    def calculate_jacob(jacobian, theta):
        jacob_values = []
        for func in jacobian:
            jacob_values.append(func(theta))
        return jacob_values

    # Calculate values for the hessian i.e. second derivatives
    @staticmethod
    def calculate_hessian(hessian, theta):
        hess_values = [[0, 0], [0, 0]]
        # Calculate the value of each function in the hessian given the values in theta
        for i in range(len(hess_values)):
            for j in range(len(hess_values)):
                func = hessian[i][j]
                hess_values[i][j] = func(theta)

        return hess_values

    # Exponential Functions

    # First derivative of the exponential function log-likelihood
    def f_exponential(self, theta):
        f_theta = np.sum(self.event) - (math.exp(theta) * np.sum(self.time))
        return f_theta

    # Second derivative of the exponential function log-likelihood
    def ff_exponential(self, theta):
        ff_theta = -math.exp(theta) * np.sum(self.time)
        return ff_theta

    # Weibull Functions

    # F1: First derivative of the weibull function log-likelihood w.r.t lambda
    def f_weibull_lambda(self, theta):
        # SUM [d - exp(B1)t^exp(B2)]
        t_exp_B2 = np.power(self.time, math.exp(theta[1]))
        f_lambda = np.sum(self.event - math.exp(theta[0]) * t_exp_B2)

        return f_lambda

    # F2: First derivative of the weibull function log-likelihood w.r.t gamma
    def f_weibull_gamma(self, theta):
        # SUM [d(ln(t)exp(B2) + 1) - t^exp(B2)ln(t)exp(B1 + B2)
        t_exp_B2 = np.power(self.time, math.exp(theta[1]))
        f_gamma = np.multiply(self.event, np.log(self.time) * math.exp(theta[1]) + 1)
        f_gamma = np.sum(f_gamma - (np.multiply(t_exp_B2, np.log(self.time)) * math.exp(theta[0] + theta[1])))

        return f_gamma

    # Second derivative of F1 w.r.t lambda
    def ff1_weibull_lambda(self, theta):
        # SUM [-t^exp(B2) * exp(B1)]
        ff1_lambda = np.sum(-np.power(self.time, math.exp(theta[1])) * math.exp(theta[0]))

        return ff1_lambda

    # Second derivative of F2 w.r.t gamma
    def ff2_weibull_gamma(self, theta):
        # SUM [(dln(t) - exp(B1)t^exp(B2)ln(t))exp(B2) - t^exp(B2)(ln(t))^2 * exp(2B2 + B1)]
        t_exp_B2 = np.power(self.time, math.exp(theta[1]))
        ln_t = np.log(self.time)
        ln2_t = (np.log(self.time)) ** 2
        ff2_gamma = (np.multiply(self.event, ln_t) - (math.exp(theta[0]) * np.multiply(t_exp_B2, ln_t))) * math.exp(
            theta[1])
        ff2_gamma = np.sum(ff2_gamma - (np.multiply(t_exp_B2, ln2_t)) * math.exp((2 * theta[1]) + theta[0]))

        return ff2_gamma

    # Second derivative of F1/F2 w.r.t gamma/lambda
    def ff_weibull_gl(self, theta):
        # SUM [-t^exp(B2)ln(t)exp(B1 + B2)]
        t_exp_B2 = np.power(self.time, math.exp(theta[1]))
        ff_gl = np.sum(np.multiply(-t_exp_B2, np.log(self.time)) * math.exp(theta[0] + theta[1]))

        return ff_gl

    # Gompertz Functions

    # F1: First and second derivative of the gompertz function log-likelihood w.r.t lambda
    def f_gompertz_lambda(self, theta):
        # d - exp(b1 - b2)*(exp(exp(b2)*t)-1)
        b1 = theta[0]
        b2 = theta[1]
        f_lam = self.event - math.exp(b1 - b2) * (np.exp(math.exp(b2) * self.time) - 1)
        f_lam = np.sum(f_lam)

        return f_lam

    # Second derivative of F1 w.r.t lambda
    def ff1_gompertz_lambda(self, theta):
        # -exp(b1 - b2)*(exp(exp(b2)*t)-1)
        b1 = theta[0]
        b2 = theta[1]
        ff1_lam = -1 * math.exp(b1 - b2) * (np.exp(math.exp(b2) * self.time) - 1)
        ff1_lam = np.sum(ff1_lam)

        return ff1_lam

    # F2: First derivative of the weibull function log-likelihood w.r.t gamma
    def f_gompertz_gamma(self, theta):
        # dtexp(b2) + exp(b1 - b2)*(exp(t*exp(b2))-1) - t*exp(t*exp(b2)+b1)
        b1 = theta[0]
        b2 = theta[1]
        f_gamma = math.exp(b2) * np.multiply(self.event, self.time) + math.exp(b1 - b2) * (
                np.exp(self.time * math.exp(b2)) - 1) - np.multiply(self.time, np.exp(self.time * math.exp(b2) + b1))

        f_gamma = np.sum(f_gamma)

        return f_gamma

    # Second derivative of F2 w.r.t gamma
    def ff2_gompertz_gamma(self, theta):
        # -t^2 * exp(t*exp(b2) + b2 + b1) + t*exp(t*exp(b2) + b1) - exp(b1 - b2)*(exp(t*exp(b2))-1) + dtexp(b2)
        b1 = theta[0]
        b2 = theta[1]
        t_2 = np.power(self.time, 2)
        ff2_gamma = -1 * np.multiply(t_2, np.exp(self.time * math.exp(b2) + b2 + b1)) + np.multiply(self.time, np.exp(
            self.time * math.exp(b2) + b1)) - math.exp(b1 - b2) * (np.exp(self.time * math.exp(b2)) - 1) + (
                            np.multiply(self.time, self.event) * math.exp(b2))
        ff2_gamma = np.sum(ff2_gamma)

        return ff2_gamma

    # Second derivative of F1/F2 w.r.t gamma/lambda
    def ff_gompertz_gl(self, theta):
        # -exp(b1 - b2) * ((t*exp(b2) - 1)*exp(t*exp(b2)) + 1)
        b1 = theta[0]
        b2 = theta[1]
        ff_gl = -1 * math.exp(b1 - b2) * ((self.time * math.exp(b2) - 1) * np.exp(self.time * math.exp(b2)) + 1)
        ff_gl = np.sum(ff_gl)

        return ff_gl
