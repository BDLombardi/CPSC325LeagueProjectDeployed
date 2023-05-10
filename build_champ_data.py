import json
import time
from riotwatcher import LolWatcher, ApiError, RiotWatcher
import random

api_key = 'Insert Key Here'  
watcher1 = LolWatcher(api_key)
my_region = 'na1'


champ_mast_labels = dict()
champs = watcher1.data_dragon.champions("13.7.1")
for champ in champs["data"].keys():
    champ_mast_labels[champ.lower()] = int(champs["data"][champ]["key"])
print(champ_mast_labels["jax"])
outfile = open("champ_labels.json","w")
json.dump(champ_mast_labels,outfile,indent=4)
outfile.close()

print("Opening Dataset")
start = time.time()
infile = open("data3.json","r")
dataset = json.load(infile)
infile.close()
end = time.time()

print(len(dataset))
print(end-start)

champ_mast_labels = dict()
champs = watcher1.data_dragon.champions("13.7.1")
for champ in champs["data"].keys():
    champ_mast_labels[champ.lower()] = int(champs["data"][champ]["key"])

print(champ_mast_labels)
#position_dict = {"TOP":0,"JUNGLE":1,"MIDDLE":2,"BOTTOM":3, "UTILITY":4}
team_list = []
y = []
for match in dataset:
    win_team = []
    lose_team = []
    bad_match = False
    if match["info"]["gameMode"]== "CLASSIC":
        for player in match["info"]["participants"]:
            if len(match["info"]["participants"]) == 10 and len(match["metadata"]["participants"]) == 10:
                if player["puuid"] == "BOT" or "challenges" not in player.keys():
                    bad_match = True
                else:
                    champ_id = champ_mast_labels[player["championName"].lower()]
                    if player["win"]==True:
                        win_team.append(champ_id)
                        if match["info"]["participants"].index(player) == 4:
                            try: 
                                win_team.append(player["challenges"]["teamBaronKills"])
                                win_team.append(player["challenges"]["teamElderDragonKills"])
                                win_team.append(player["challenges"]["teamRiftHeraldKills"])
                            except:
                                print(1)
                                outfile = open("match.json","w")
                                json.dump(match,outfile,indent=4)
                                outfile.close()
                                exit()
                    else:
                        lose_team.append(champ_id)
                        if match["info"]["participants"].index(player)  == 9:
                            try:
                                lose_team.append(player["challenges"]["teamBaronKills"])
                                lose_team.append(player["challenges"]["teamElderDragonKills"])
                                lose_team.append(player["challenges"]["teamRiftHeraldKills"])
                            except:
                                print(2)
                                outfile = open("match.json","w")
                                json.dump(match,outfile,indent=4)
                                outfile.close()
                                exit()
            else: 
                bad_match = True
        if bad_match == False:
            swap = random.randint(0,1)
            if swap ==0:
                win_team.extend(lose_team)
                team_list.append(win_team)
            else: 
                lose_team.extend(win_team)
                team_list.append(lose_team)
            y.append(swap)
    else:
        pass


print("Success, writing data ")
outfile = open("champ_data4.json","w")
json.dump(team_list,outfile,indent=4)
outfile.close()
outfile = open("win_data4.json","w")
json.dump(y,outfile,indent=4)
outfile.close()

