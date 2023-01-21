import numpy as np
import pandas as pd
import math


class Kaplan_Meier:
    """"
    Class for fitting Kaplan-Meier estimates of the survival function

    Attributes:
        events: Array of the events observed
        times: Array of the times the events were observed
        sample_size: Int referring to how many observations are in the dataset
        survival_data: Pandas Data Frame of the survival estimates calculated using Kaplan-Meier
        covariates: Any covariates in the dataset for fitting two kaplan meiers
        CI: Boolean variable, whether confidence intervals should be calculated or not

    """

    def __init__(self):
        # Initiate the attributes and create table of survival function estimates
        self.data = None
        self.times = None
        self.events = None
        self.ci = None
        self.survival_data = None

        self.failure_count = None
        self.censored_count = None


    # Fit the Kaplan-Meier model to two binary groups e.g. treatment and control
    def fit_kaplan_meier_two_groups(self, data, covariate, CI=True):
        
        # Check len(Time) == len(Event)
        if(len(data['Time']) != len(data['Event'])):
           print('Time and Event columns must have equal length')
           exit()
        
        # Check there are no 0 or negative times
        if((data['Time']<0).values.any() or (data['Time']==0).values.any()):
           print('Times must be > 0')
           exit()

        # Split the data into the two binary groups - using covariate where covariate = 0 and covariate = 1
        base_group_index, = np.where(data[covariate] == 0)
        base_group_data = data.iloc[base_group_index]
        base_surv = self.fit_kaplan_meier(data=base_group_data, CI=CI)
        base_cov_value = np.full(len(base_surv), 0)
        base_surv[covariate] = base_cov_value

        pos_group_index, = np.where(data[covariate] == 1)
        pos_group_data = data.iloc[pos_group_index]
        pos_surv = self.fit_kaplan_meier(data=pos_group_data, CI=CI)
        pos_cov_value = np.full(len(pos_surv), 1)
        pos_surv[covariate] = pos_cov_value

        return base_surv, pos_surv

    # Fit the Kaplan-Meier survival function estimate and return data frame
    def fit_kaplan_meier(self, data, CI=True):
        
        # Check len(Time) == len(Event)
        if(len(data['Time']) != len(data['Event'])):
           print('Time and Event columns must have equal length')
           exit()
        
        # Check there are no 0 or negative times
        if((data['Time']<0).values.any() or (data['Time']==0).values.any()):
           print('Times must be > 0')
           exit()
        
        self.data = data
        self.times = self.data['Time']
        self.events = self.data['Event']
        self.ci = CI

        # Obtain the unique sorted failure times and add in time 0
        unique_times = self.get_unique_times()

        # Get the failure and censor count for each unique failure time
        self.event_count(unique_times)

        # Get the risk set for each unique failure time
        risk_set = self.get_risk_set(unique_times)

        # Get the kaplan meier estimates for survival
        km_estimates = self.kaplan_meier_estimates(unique_times, risk_set)

        # Create pandas dataframe of output
        survival_data = {'Time': unique_times,
                         'Risk Set': risk_set,
                         'S(t)': km_estimates}

        self.survival_data = pd.DataFrame(survival_data)

        if self.ci is True:
            self.calculate_CIs()

        return self.survival_data  # Return the survival data frame

    # Obtain the unique sorted failure times and add in time 0
    def get_unique_times(self):

        # Only have times where the event happened i.e. event == 1
        indexes, = np.where(self.events == 1)
        fail_times = self.times.copy()
        fail_times = np.unique(fail_times.iloc[indexes])

        # If time 0 is not in the unique failure times, add it in
        if 0 not in fail_times:
            fail_times = np.append(fail_times, 0)

        return np.sort(fail_times)

    # Get the number of censoring/failures in each time point - returns two arrays
    def event_count(self, unique_times):

        # Get failure count for each fail time
        failure_count = []  # Array of the failure counts for each corresponding unique failure time
        for t in unique_times:
            fc = np.count_nonzero((self.times == t) & (self.events == 1))
            failure_count.append(fc)

        self.failure_count = failure_count

        # Get the censored count for each fail time
        censored_count = []  # Array of the censored count for each corresponding unique failure time
        for j in range(len(unique_times)):
            cc = 0  # Censored count for unique_times[j]
            # If it is the last unique failure time in the array
            if j == len(unique_times) - 1:
                cc = np.count_nonzero((self.times >= unique_times[j]) & (self.events == 0))
            else:
                cc = np.count_nonzero((unique_times[j] <= self.times) & (self.times < unique_times[j + 1]) & (self.events == 0))
            censored_count.append(cc)

        self.censored_count = censored_count

    # Calculate the risk set for each unique failure time t: sample_size - (No. failed at time t-1 + No. censored at
    # time t-1)
    def get_risk_set(self, unique_times):
        sample_size = len(self.times)
        risk_set = []  # Array of the risk set in each corresponding unique failure time
        failure_plus_censored = self.failure_count + self.censored_count

        for i in range(len(unique_times)):
            if i == 0:  # At time = 0, the risk set is the whole sample size
                risk_set.append(sample_size)
            else:
                #rs = risk_set[i - 1] - failure_plus_censored[i-1]
                rs = risk_set[i-1] - failure_plus_censored[i]
                risk_set.append(rs)

        return risk_set  # Return the risk set array

    # Calculates the kaplan_meier estimate for each unique time point - returns array
    def kaplan_meier_estimates(self, unique_times, risk_set):
        km_estimates = []  # Array of kaplan-meier survival function estimates for each unique failure time
        risk_set = np.asarray(risk_set)
        failure_count = np.asarray(self.failure_count)

        km_estimates = np.true_divide((risk_set - self.failure_count), risk_set)

        # Calculate the overall S(t) estimate for each time point - probability of surviving up to time t
        overall_st = []
        for j in range(len(unique_times)):
            if j == 0:
                st = 1  # First one must be = 1 (no one has had the event yet)
            else:
                st = km_estimates[j] * overall_st[j - 1]
            overall_st.append(st)

        return overall_st  # Return the array of kaplan-meier S(t) estimates

    # Method to calculate SE using Greenwood's method - takes a df containing survival data
    def calculate_SE(self):
        d = self.failure_count
        n = self.survival_data['Risk Set']
        st = self.survival_data['S(t)']

        # Calculate the standard errors
        ses = []
        for j in range(len(st)):
            s_hat_t = st[j]
            sum = 0
            i = 0
            while i < j:
                sum = sum + d[i] / (n[i] * (n[i] - d[i]))
                i += 1
            se_st = s_hat_t * math.sqrt(sum)
            ses.append(se_st)

        return ses

    # Method to calculate the CIs and append to dataframe
    def calculate_CIs(self):

        # Calculate the upper and lower confidence intervals
        st = self.survival_data['S(t)']
        ses = self.calculate_SE()

        uci = st + (1.96 * np.multiply(ses, st))
        lci = st - (1.96 * np.multiply(ses, st))

        self.survival_data['Upper CI'] = uci
        self.survival_data['Lower CI'] = lci

        return self.survival_data
