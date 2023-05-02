import streamlit as st
from riotwatcher import LolWatcher, ApiError, RiotWatcher
import numpy as np
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import base64
import matplotlib.pyplot as plt
import json

st.write("""
        # CPSC 325 League of Legends Dashboard and Predictor
        Welcome to the League of Legends Player Dashboard and Predictor! 

        This website is designed to help players visualize their own account data, as well as offer various ML models to help predict the winners of games. 
         """)


options = st.selectbox('Choose what endpoint that you want to engage with.', ["Account Summary","Predictors"])
st.write(options)

if options == "Account Summary":
    st.write("Please enter a Account name on the North American Server")
    text_input = st.text_input('Account Name')
    if text_input != "":
        username = text_input
        api_key = 'RGAPI-7c56ac6e-237c-476b-9ca9-4ee2fe089ca4'  
        watcher1 = LolWatcher(api_key)
        my_region = 'na1'

        me = watcher1.summoner.by_name(my_region, username)
        game_count = 20
        my_match_ids = watcher1.match.matchlist_by_puuid(my_region, me['puuid'],count=game_count)
        data = []
        for entry in my_match_ids:
            data.append(watcher1.match.by_id(my_region,entry))
        total_kills = []
        total_deaths = []
        total_dam = []
        total_wards = []
        total_assists = []
        dam_taken = []
        gold = []
        level = []
        gold_spent = []
        for entry in data:
            idx = entry["metadata"]["participants"].index(me["puuid"])
            total_kills.append(entry["info"]["participants"][idx]["kills"])
            total_deaths.append(entry["info"]["participants"][idx]["deaths"])
            total_dam.append(entry["info"]["participants"][idx]["totalDamageDealtToChampions"])
            total_wards.append(entry["info"]["participants"][idx]["wardsPlaced"])
            total_assists.append(entry["info"]["participants"][idx]["assists"])
            dam_taken.append(entry["info"]["participants"][idx]["totalDamageTaken"])
            gold.append(entry["info"]["participants"][idx]["goldEarned"])
            level.append(entry["info"]["participants"][idx]["champLevel"])
            gold_spent.append(entry["info"]["participants"][idx]["goldSpent"])
        
        champ_mast = watcher1.champion_mastery.by_summoner(my_region,me["id"])
        champs = watcher1.data_dragon.champions("13.7.1")
        champ_mast_labels = dict()

        for champ in champs["data"].keys():
            champ_mast_labels[champ] = int(champs["data"][champ]["key"])

        champ_mast_labels_rev = dict()
        for name in champ_mast_labels:
            champ_mast_labels_rev[champ_mast_labels[name]] = name

        mast_scores = []
        champ_levels = []
        labels = []
        for entry in champ_mast:
            mast_scores.append(entry["championPoints"])
            champ_levels.append(entry["championLevel"])
            labels.append(champ_mast_labels_rev[entry["championId"]])

        df = pd.DataFrame({
            'x': mast_scores,
            'y': champ_levels,
            's': mast_scores,
            'group': labels
        })
        fig,ax = plt.subplots(figsize=(20,10))
        ax = sns.scatterplot(data = df, x =df.x,y= df.y, alpha = 0.5,s = df.s)

        for line in range(0,df.shape[0]):
            ax.text(df.x[line], df.y[line], df.group[line], horizontalalignment='center', size='medium', color='black', weight='semibold')
        st.pyplot(fig)

        fig,ax = plt.subplots(2,2)
        game_count_list = [ i for i in range(game_count)]
        ax[0,0].plot(game_count_list,total_kills)
        ax[0,0].plot(game_count_list,total_deaths)
        ax[0,0].plot(game_count_list,total_assists)
        ax[0,0].plot(game_count_list,total_wards)
        ax[0,0].legend(["Kills","Deaths","Assists","Wards"])


        ax[0,1].plot(game_count_list,total_dam)
        ax[0,1].plot(game_count_list,dam_taken)
        ax[0,1].legend(["Damage Dealt to Champions", "Damage Taken"])


        ax[1,0].plot(game_count_list,gold)
        ax[1,0].plot(game_count_list,gold_spent)
        ax[1,0].legend(["Gold"])

        ax[1,1].plot(game_count_list,level)
        ax[1,1].legend(["Champion Level"])

        st.pyplot(fig)
elif options == "Predictors":
    model_options = st.selectbox('Please select a model to use. ', ["Team Comp + Baron","Team Comp + Dragon", "Team Comp + First Blood", "Team Comp + Rift Herald", "Objectives"])
    st.write(model_options)
    infile = open("champ_labels.json","r")
    champ_labels = json.load(infile)
    infile.close()
    if model_options == "Team Comp + Baron":
        top = st.text_input("Top Laner")
        st.write(champ_labels[top])
        
