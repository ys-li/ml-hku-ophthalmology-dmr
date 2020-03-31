import pandas as pd
import os
import re

directory = 'datasets/HKU'
temp_arr = []

for filename in os.listdir(directory):
    print(f"{filename}")
    matches = re.findall(r'ER(.*?)M', filename)
    temp_arr.append([f'datasets/HKU/{filename}/RE01.png', float(matches[0])])
    temp_arr.append([f'datasets/HKU/{filename}/LE01.png', float(matches[1])])

df = pd.DataFrame(temp_arr, columns=['path', 'diagnosis'])
df.to_csv("hku.csv")