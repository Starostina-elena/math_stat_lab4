import numpy as np
from scipy.stats import f

price_range = []
battery = []

with open('mobile_phones.csv') as file:
    lines = file.readlines()
    for i in lines[1:]:
        line = i.strip().split(',')
        price_range.append(int(line[-1]))
        battery.append(int(line[0]))

price_range = np.array(price_range)
battery = np.array(battery)

groups = [battery[price_range == level] for level in np.unique(price_range)]

overall_mean = np.mean(battery)

ssb = sum(len(group) * (np.mean(group) - overall_mean) ** 2 for group in groups)
ssw = sum(np.sum((group - np.mean(group)) ** 2) for group in groups)

df_between = len(groups) - 1
df_within = len(battery) - len(groups)

msb = ssb / df_between
msw = ssw / df_within

f_stat = msb / msw

p_value = f.sf(f_stat, df_between, df_within)

print("f-статистика:", f_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Отвергнуть H0")
else:
    print("Принять H0")