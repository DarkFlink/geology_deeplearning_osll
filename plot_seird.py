import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from calc_params import params

day_step = 4
days = np.arange(0, params['days_from_start'], day_step)
start_date = datetime.strptime('02-04-2020', '%d-%m-%Y')
dates = tuple(start_date + timedelta(days=i*day_step) for i in range(len(days)))
dates_str = tuple(map(lambda date: date.strftime('%Y-%m-%d'), dates))


def plotseird(t, S, E, I, R, D):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    #ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Подозреваемых')
    #ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Зараженных')
    # ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Умерших')
    # ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')

    ax.set_xlabel('Время (дней). 0-й день - 02.04.2020')
    ax.set_ylabel('Кол-во человек')

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.grid(color='black', linestyle='-', linewidth=1)

    plt.xticks(days, labels=dates_str, rotation=25)
    plt.yticks(np.arange(0, 150, step=10))
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.grid(which='minor', color='#CCCCCC', linestyle=':')
    plt.legend(loc='upper left')

    plt.show()
