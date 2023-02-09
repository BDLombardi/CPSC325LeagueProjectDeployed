import json 
import pandas as pd

file = open("hist.json")
data = json.load(file)

puuid = "3stDfZ0IZqoKrP3sgZmRc3VSOhd9BqR0N9Uv2v4ji7vNbUGce9cnqvGbGKpMU7dpJ2uRKSjScBtPBw"
print(data[0]["metadata"]["participants"].index(puuid))
print(data[1]["info"]["participants"][data[1]["metadata"]["participants"].index(puuid)]["puuid"])


total_kills = 0
total_deaths = 0
total_dam = 0
total_vision = 0
for entry in data:
    idx = entry["metadata"]["participants"].index(puuid)
    total_kills += entry["info"]["participants"][idx]["kills"]
    total_deaths += entry["info"]["participants"][idx]["deaths"]
    total_dam += entry["info"]["participants"][idx]["totalDamageDealtToChampions"]
    total_vision += entry["info"]["participants"][idx]["visionScore"]

print("Total kills: ", total_kills)
print("Average kills: ", total_kills/ len(data))
print("Total deaths: ", total_deaths)
print("Average deaths: ", total_deaths/ len(data))
print("Average Total Damage: ", total_dam/len(data))
print("Average Vision Score: ", total_vision/len(data))
file.close()