import streamlit as st
from riotwatcher import LolWatcher, ApiError, RiotWatcher
import numpy as np
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import base64
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import joblib
import sklearn
from seaborn import objects as so

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
        api_key = st.secrets['KEY']
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
        champs_played = dict()
        runes_used = dict()
        roles_played = {"TOP": 0, "JUNGLE": 0, "MIDDLE": 0,"BOTTOM": 0, "UTILITY": 0}
        infile = open("rune_dict.json","r")
        rune_keys = json.load(infile)
        infile.close()
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
            roles_played[entry["info"]["participants"][idx]["individualPosition"]] +=1
            if entry["info"]["participants"][idx]["championName"] in champs_played.keys():
                champs_played[entry["info"]["participants"][idx]["championName"]] +=1
            else: 
                champs_played[entry["info"]["participants"][idx]["championName"]] =1
            if rune_keys[str(entry["info"]["participants"][idx]["perks"]["styles"]["selections"][0]["perk"])] in runes_used.keys():
                runes_used[rune_keys[str(entry["info"]["participants"][idx]["perks"]["styles"][0]["selections"][0]["perk"])]] +=1
            else: 
                runes_used[rune_keys[str(entry["info"]["participants"][idx]["perks"]["styles"][0]["selections"][0]["perk"])]] =1
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
            mast_scores.append(entry["championPoints"] + np.random.rand(1)[0] * 4000)
            champ_levels.append(entry["championLevel"] + np.random.rand(1)[0] * 0.4)
            labels.append(champ_mast_labels_rev[entry["championId"]])

        df = pd.DataFrame({
            'x': mast_scores,
            'y': champ_levels,
            's': mast_scores,
            'champions': labels
        })
        fig,ax = plt.subplots(figsize=(20,10))
        ax = sns.scatterplot(data = df, x =df.x[0:25],y= df.y[0:25], alpha = 0.5,s = df.s[0:25],hue = df.champions[0:25])

        for line in range(0,25):
            ax.text(df.x[line], df.y[line], df.champions[line], horizontalalignment='center', size='medium', color='black', weight='semibold')
        ax.set_title("Top 25 Mastery Champions Bubble Chart")
        st.pyplot(fig)

        fig,ax = plt.subplots(2,2,figsize=(10,10))
        game_count_list = [ i for i in range(game_count)]
        ax[0,0].plot(game_count_list,total_kills)
        ax[0,0].plot(game_count_list,total_deaths)
        ax[0,0].plot(game_count_list,total_assists)
        ax[0,0].plot(game_count_list,total_wards)
        ax[0,0].legend(["Kills","Deaths","Assists","Wards"])
        ax[0,0].set_xlabel("Game Number")
        ax[0,0].set_ylabel("Count")


        ax[0,1].plot(game_count_list,total_dam)
        ax[0,1].plot(game_count_list,dam_taken)
        ax[0,1].legend(["Damage Dealt to Champions", "Damage Taken"])
        ax[0,1].set_xlabel("Game Number")
        ax[0,1].set_ylabel("Count")


        ax[1,0].plot(game_count_list,gold)
        ax[1,0].plot(game_count_list,gold_spent)
        ax[1,0].legend(["Gold","Gold Spent"])
        ax[1,0].set_xlabel("Game Number")
        ax[1,0].set_ylabel("Count")

        ax[1,1].plot(game_count_list,level)
        ax[1,1].legend(["Champion Level"])
        ax[1,1].set_xlabel("Game Number")
        ax[1,1].set_ylabel("Count")

        st.pyplot(fig)

        fig,ax = plt.subplots(2,figsize=(15,15))
        ax[0].bar(range(len(champs_played)),list(champs_played.values()),tick_label=list(champs_played.keys()))
        ax[1].bar(range(len(roles_played)),list(roles_played.values()),tick_label=list(roles_played.keys()))
        st.pyplot(fig)

        fig,ax = plt.subplots(1,figsize=(20,10))
        ax[0].bar(range(len(runes_used)),list(runes_used.values()),tick_labels=list(runes_used.keys()))
        st.pyplot(fig)


elif options == "Predictors":
    model_options = st.selectbox('Please select a model to use. ', ["Team Comp + Baron","Team Comp + Dragon", "Team Comp + Rift Herald", "Objectives"])
    st.write(model_options)
    infile = open("champ_labels.json","r")
    champ_labels = json.load(infile)
    infile.close()
    if model_options == "Team Comp + Baron":
        st.write("Team 1:")
        top1 = st.selectbox("Team 1: Top Laner",champ_labels.keys())
        jg1 = st.selectbox("Team 1: Jungle",champ_labels.keys())
        mid1 = st.selectbox("Team 1: Mid",champ_labels.keys())
        bot1 = st.selectbox("Team 1: Bot",champ_labels.keys())
        sup1 = st.selectbox("Team 1: Support",champ_labels.keys())
        b_1 = int(st.selectbox("Team 1 Baron First", [True,False]))

        st.write("Team 2:")
        top2 = st.selectbox("Team 2: Top Laner",champ_labels.keys())
        jg2 = st.selectbox("Team 2: Jungle",champ_labels.keys())
        mid2 = st.selectbox("Team 2: Mid",champ_labels.keys())
        bot2 = st.selectbox("Team 2: Bot",champ_labels.keys())
        sup2 = st.selectbox("Team 2: Support",champ_labels.keys())
        b_2 = int(st.selectbox("Team 2 Baron First", [True,False]))
        X_test = [[champ_labels[top1],champ_labels[jg1],champ_labels[mid1],champ_labels[bot1],champ_labels[sup1],b_1,champ_labels[top2],champ_labels[jg2],champ_labels[mid2],champ_labels[bot2], champ_labels[sup2],b_2]]
        comp_bf = tf.keras.models.load_model('pred_models/comp_bf/143-0.5252.hdf5')
        y_pred = comp_bf.predict(X_test)
        st.write("Predicted Winning Team: ", np.argmax(y_pred)+1)
    elif model_options == "Team Comp + Dragon":
        st.write("Model Accuracy on Test Set: 63.9%")
        st.write("Team 1:")
        top1 = st.selectbox("Team 1: Top Laner",champ_labels.keys())
        jg1 = st.selectbox("Team 1: Jungle",champ_labels.keys())
        mid1 = st.selectbox("Team 1: Mid",champ_labels.keys())
        bot1 = st.selectbox("Team 1: Bot",champ_labels.keys())
        sup1 = st.selectbox("Team 1: Support",champ_labels.keys())
        b_1 = int(st.selectbox("Team 1 Dragon First", [True,False]))

        st.write("Team 2:")
        top2 = st.selectbox("Team 2: Top Laner",champ_labels.keys())
        jg2 = st.selectbox("Team 2: Jungle",champ_labels.keys())
        mid2 = st.selectbox("Team 2: Mid",champ_labels.keys())
        bot2 = st.selectbox("Team 2: Bot",champ_labels.keys())
        sup2 = st.selectbox("Team 2: Support",champ_labels.keys())
        b_2 = int(st.selectbox("Team 2 Dragon First", [True,False]))
        X_test = [[champ_labels[top1],champ_labels[jg1],champ_labels[mid1],champ_labels[bot1],champ_labels[sup1],b_1,champ_labels[top2],champ_labels[jg2],champ_labels[mid2],champ_labels[bot2], champ_labels[sup2],b_2]]
        comp_df = tf.keras.models.load_model('pred_models/comp_dragon/64-0.6470.hdf5')
        y_pred = comp_df.predict(X_test)
        st.write("Predicted Winning Team: ", np.argmax(y_pred)+1)

    elif model_options == "Team Comp + Rift Herald":
        st.write("Model Accuracy on Test Set: 57.6%")
        st.write("Team 1:")
        top1 = st.selectbox("Team 1: Top Laner",champ_labels.keys())
        jg1 = st.selectbox("Team 1: Jungle",champ_labels.keys())
        mid1 = st.selectbox("Team 1: Mid",champ_labels.keys())
        bot1 = st.selectbox("Team 1: Bot",champ_labels.keys())
        sup1 = st.selectbox("Team 1: Support",champ_labels.keys())
        b_1 = int(st.selectbox("Team 1 Rift Herald First", [True,False]))

        st.write("Team 2:")
        top2 = st.selectbox("Team 2: Top Laner",champ_labels.keys())
        jg2 = st.selectbox("Team 2: Jungle",champ_labels.keys())
        mid2 = st.selectbox("Team 2: Mid",champ_labels.keys())
        bot2 = st.selectbox("Team 2: Bot",champ_labels.keys())
        sup2 = st.selectbox("Team 2: Support",champ_labels.keys())
        b_2 = int(st.selectbox("Team 2 Rift Herald First", [True,False]))
        X_test = [[champ_labels[top1],champ_labels[jg1],champ_labels[mid1],champ_labels[bot1],champ_labels[sup1],b_1,champ_labels[top2],champ_labels[jg2],champ_labels[mid2],champ_labels[bot2], champ_labels[sup2],b_2]]
        comp_fb = joblib.load('pred_models/comp_rf/forest_clf.pkl')
        y_pred = comp_fb.predict(X_test)
        st.write("Predicted Winning Team: ", np.argmax(y_pred)+1)

    elif model_options == "Objectives":
        st.write("Model Accuracy on Test Set: 86.7%")
        st.write("Team 1:")
        b_1 = int(st.selectbox("Team 1 Baron First", [True,False]))
        d_1 = int(st.selectbox("Team 1 Dragon First", [True,False]))
        c_1 = int(st.selectbox("Team 1 First Blood", [True,False]))
        i_1 = int(st.selectbox("Team 1 Inhibitor First", [True,False]))
        h_1 = int(st.selectbox("Team 1 Rift Herald First", [True,False]))
        t_1 = int(st.selectbox("Team 1 Tower First", [True,False]))

        st.write("Team 2:")
        b_2 = int(st.selectbox("Team 2 Baron First", [True,False]))
        d_2 = int(st.selectbox("Team 2 Dragon First", [True,False]))
        c_2 = int(st.selectbox("Team 2 First Blood", [True,False]))
        i_2 = int(st.selectbox("Team 2 Inhibitor First", [True,False]))
        h_2 = int(st.selectbox("Team 2 Rift Herald First", [True,False]))
        t_2 = int(st.selectbox("Team 2 Tower First", [True,False]))

        X_test = [[b_1,d_1,c_1,i_1,h_1,t_1,b_2,d_2,c_2,i_2,h_2,t_2]]
        obj_mod = tf.keras.models.load_model('pred_models/obj/121-0.2676.hdf5')
        y_pred = obj_mod.predict(X_test)
        st.write("Predicted Winning Team: ", np.argmax(y_pred)+1)

        
