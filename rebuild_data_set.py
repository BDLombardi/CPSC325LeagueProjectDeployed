import json 
from riotwatcher import LolWatcher, ApiError, RiotWatcher
import copy



api_key = 'Insert Key Here'
watcher1 = LolWatcher(api_key)
my_region = 'na1'

infile = open("annotations.json","r")
annotations = json.load(infile)
infile.close()
match_dict = annotations["matches"]

data= []

i = 0
for key in match_dict:
    if i % 500 == 0:
        print(i)
    if i % 10000 ==0 and i !=0:
        print("Saving")
        outfile = open("data_2.json","w")
        json.dump(data,outfile)
        outfile.close()
    try:
        data.append(watcher1.match.by_id(my_region,key))
    except:
        print("Failed, writing data")
        outfile = open("data_2.json","w")
        json.dump(data,outfile,indent=4)
        outfile.close()
        break
    i +=1

print("Success, writing data ")
outfile = open("data_2.json","w")
json.dump(data,outfile)
outfile.close()

