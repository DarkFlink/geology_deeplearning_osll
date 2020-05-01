import SEIR
import pandas
import numpy as np
from scipy.integrate import odeint
import csv
import json

import calc_params
import plot_seird

# indeces
index_infected = 1
index_recovered = 2
index_dead = 3
index_selfiso = 7

# params from json
params = {}
with open('params.json') as json_file:
    params = json.load(json_file)

# open csv with stats
# don't see on nans on date field, we don't need dates  basically
path = 'spb_covid-19.csv'
spb_data = np.genfromtxt(path, delimiter=',', dtype=int)
spb_data = spb_data.transpose()

# processed contants
N = params['population']  # SPB population
alpha = calc_params.death_rate(spb_data[index_recovered], spb_data[index_dead])
R_0 = params['amount_of_people_an_infected_person_infects'] / calc_params.self_isolation_index(spb_data[index_selfiso]) \
      * params['self_isolation_influence_factor']
delta = 1.0 / params['incubation_period']  # incubation period
D = params['incubation_period']  # the worst case, when person infects each other in full time period of incubation
gamma = 1.0 / D
beta = R_0 * gamma  # R_0 = beta / gamma, so beta = R_0 * gamma
rho = 1 / params['period_until_death']  # 9 days from infection until death

S0, E0, I0, R0, D0 = N-1, spb_data[index_infected][0], \
                     spb_data[index_infected][0], \
                     spb_data[index_recovered][0], \
                     spb_data[index_dead][0]  # initial conditions: one exposed
t = np.linspace(0, params['days_from_start'])  # Grid of time points (in days)
y0 = S0, E0, I0, R0, D0  # Initial conditions vector

def deriv(y, t, N, beta, gamma, delta, alpha, rho):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * rho * I
    return dSdt, dEdt, dIdt, dRdt, dDdt

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))
S, E, I, R, D = ret.T

# plot
plot_seird.plotseird(t, S, E, I, R, D)