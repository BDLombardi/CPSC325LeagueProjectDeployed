import json 
from riotwatcher import LolWatcher, ApiError, RiotWatcher
import copy


api_key = 'Insert Keye Here'
watcher1 = LolWatcher(api_key)
my_region = 'na1'

# infile = open("data.json","r")
# dataset = json.load(infile)
# infile.close()

infile = open("annotations.json","r")
annotations = json.load(infile)
infile.close()
match_dict = annotations["matches"]
name_dict = annotations["names"]

i = 0

stable_names = copy.deepcopy(list(name_dict.keys()))
stable_names = stable_names[-1000:]

dataset = []
for key in stable_names:
    my_match_ids = watcher1.match.matchlist_by_puuid(my_region, key ,count=100)
    data = []
    for entry in my_match_ids: 
        data.append(watcher1.match.by_id(my_region,entry))
    for entry in data:
        try:
            if match_dict[entry["metadata"]["matchId"]] == True:
                pass
        except:
            match_dict[entry["metadata"]["matchId"]] = True
            dataset.append(entry)
        for name in entry["metadata"]["participants"]:
            try:
                if name_dict[name] == True:
                    pass
            except:
                name_dict[name] = True
    print(len(match_dict))

    print(len(name_dict))

    annotations = dict()
    annotations["names"] = name_dict
    annotations["matches"] = match_dict
    outfile = open("annotations.json","w")
    json.dump(annotations,outfile,indent=4)
    outfile.close()

    outfile = open("data3.json","w")
    json.dump(dataset,outfile,indent=4)
    outfile.close()
    i +=1 
    print(i)
    print(len(dataset))
    if i == 1000: 
        break

# print(len(match_dict))

# print(len(name_dict))

# annotations = dict()
# annotations["names"] = name_dict
# annotations["matches"] = match_dict
# outfile = open("annotations.json","w")
# json.dump(annotations,outfile)
# outfile.close()

# outfile = open("data.json","w")
# json.dump(dataset,outfile)
# outfile.close()

