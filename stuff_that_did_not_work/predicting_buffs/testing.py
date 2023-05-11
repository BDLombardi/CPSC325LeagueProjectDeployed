import json
import pandas as pd
import os

df = pd.read_csv("winrates_csv\League of Legends Champion Stats 12.1.csv")
df_buffs = pd.read_csv("buffs.csv",header=0)

files = os.listdir("winrates_csv")

x = []
y = []

for entry in files: 
    df = pd.read_csv("winrates_csv/" + entry,header=0)
    patch = df_buffs.iloc[files.index(entry)]
    for i in range(len(df)):
        temp = [float(df.iloc[i]["Trend"]),float(df.iloc[i]["KDA"]),float(df.iloc[i]["Win %"][:-1]),float(df.iloc[i]["Pick %"][:-1]),float(df.iloc[i]["Role %"][:-1]),float(df.iloc[i]["Ban %"][:-1])]
        x.append(temp)
        if df.iloc[i]["Name"] in patch["Nerfs"]:
            y.append(1)
        elif df.iloc[i]["Name"] in patch["Buffs"]:
            y.append(2)
        else: 
            y.append(0)

outfile = open("buff_x_train.json","w")
json.dump(x,outfile,indent=4)
outfile.close()

outfile = open("buff_y_train.json","w")
json.dump(y,outfile,indent=4)
outfile.close()

