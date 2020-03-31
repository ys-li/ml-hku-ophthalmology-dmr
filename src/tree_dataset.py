import pandas as pd
import sys
import os
import shutil
csv_name = sys.argv[1]
images_path = sys.argv[2]
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def set_name_and_copy(df, name, folder_name):
    mkdir(f'{images_path}/{folder_name}')
    #shutil.copyfile(f'{images_path}/{name}.jpg', f'{images_path}/{folder_name}/{name}.jpg')
    df.loc[df['file_name'] == name, 'file_name'] = f'{folder_name}/{name}'
df = pd.read_csv(csv_name)
subset = "abcdefghijklmnopqrstuvwxyz"
i = 0
for row in df.iterrows():
    file_name = row[1]['file_name']
    folder_name = subset[int(i / 3000)]
    set_name_and_copy(df, file_name, folder_name)
    i += 1
df.to_csv(f'{csv_name}_new')
    

