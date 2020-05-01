# history_smooth_gap - колво измерений из истории, которое будет учитываться
# при расчете вероятности смерти (т.е N последних замеров)
import json

params = {}
with open('params.json') as json_file:
    params = json.load(json_file)

history_smooth_gap=params['history_smooth_gap']

def death_rate(recovered, deceased):
    rate = 0
    for i in range(len(recovered)-history_smooth_gap, len(recovered)):
        rate += deceased[i]/recovered[i]
    return rate/history_smooth_gap


def self_isolation_index(self_is_his):
    rate = 0
    for i in range(len(self_is_his)-history_smooth_gap, len(self_is_his)):
        rate += self_is_his[i]
    return rate/history_smooth_gap/2.5
