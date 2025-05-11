import numpy as np
from scipy.stats import f_oneway

price_range = []
battery = []

with open('mobile_phones.csv') as f:
    lines = f.readlines()
    for i in lines[1:]:
        line = i.strip().split(',')
        price_range.append(int(line[-1]))
        battery.append(int(line[0]))

price_range = np.array(price_range)
battery = np.array(battery)

groups = [battery[price_range == level] for level in np.unique(price_range)]

f_stat, p_value = f_oneway(*groups)

print("F-statistic:", f_stat)
print("P-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between group means.")
else:
    print("Fail to reject the null hypothesis: No significant difference between group means.")