# CPSC 325 League of Legends Project 
### By Ben Lombardi


Welcome to my CPSC 325 Data Science Project on League of Legends. This project is deployed on [Streamlit](https://bdlombardi-cpsc325leagueprojectdeployed-app-7btdsr.streamlit.app/). 

The goal of this project was to develop and deploy an account level dashboard that displayed summary statistics and visualizations for accounts on the North American Server. Additionally this app has several Machine Learning models that can be used to predict the winner of games based on Team Composition and which team is the first to take and objective.

Below is a description of important scripts.

## Important Scripts

* app.py - this is what Streamlit uses to run the app 
* pred_models - this folder contains all prediction models for the web-app. 
* build_champ.ipynb - this is where almost all of the later-stage training and feature selection took place. One has to manually select(or slice) which data they want for X and y in this notebook as well as adjust the input features of the neural network to correspond. This was used in an ipynb file as it was easier to load and keep the dataset loaded in an ipynb file as it normally took 3 minutes to load in with json. 
* rebuild_data_set.py - This is the final iteration of the script that was used to build the dataset. 
* rune_dict.json - simplified dictionary of runes that is used for app.py  
* champ_labels.json - a dictionary of every champions id that is used for app.py
* training.py - an early iteration of my training script before I moved over to an ipynb file
* champ_data3.json and champ_data4.json - early iterations and practices for X dataset for training. 
* build_data_set.py - early iteration of rebuild_data_set.py 
* build_champ_data.py - early iteration of assembling X data. 
* experimenting.ipynb - notebook used to help debug the streamlit app.  
* in stuff_that_did_not_work 
    * testing.py - used to build dataset for experimenting to predict buffs and nerfs 
    * winrates_csv - contains all winrates per patch in season 12
    * buff_training.py - initial script for training model 
    * buff_window.ipynb - script used for EDA and model training 
    * buffs.csv - csv containing all buffs and nerfs for the year

