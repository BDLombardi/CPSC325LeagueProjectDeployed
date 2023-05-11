import tensorflow as tf
import json
import random 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from riotwatcher import LolWatcher, ApiError, RiotWatcher
import numpy as np
import sklearn
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import random
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def build_training_set(data):
    api_key = 'Insert Key Here'  
    watcher1 = LolWatcher(api_key)
    my_region = 'na1'
    role_class = {"Assassin": 0, "Fighter": 1, "Mage": 2, "Marksman": 3, "Support": 4, "Tank": 5}
    champ_mast_labels = dict()
    champs = watcher1.data_dragon.champions("13.7.1")
    for champ in champs["data"].keys():
        champ_mast_labels[champ] = int(champs["data"][champ]["key"])
    
    for champ in champ_mast_labels.keys():
        if champ_mast_labels[champ] == 0:
            print(champ)

    champ_mast_labels_rev = dict()
    for name in champ_mast_labels:
        champ_mast_labels_rev[champ_mast_labels[name]] = name
    
    y = []
    data1 = []
    for entry in data:
        swap = random.randint(0,1)
        team1 = list(entry[0:5])
        team2 = list(entry[5:])
        roles = []
        temp_list = []
        if swap == 1:
            team2.extend(team1)
            temp_list.extend(team2)
        else: 
            team1.extend(team2)
            temp_list.extend(team1)
        roles = []
        for item in temp_list:
            roles.append(role_class[champs["data"][champ_mast_labels_rev[item]]["tags"][0]])
        temp_list.extend(roles)
        if len(temp_list) !=20:
            pass
        else:
            data1.append(temp_list)
            y.append(swap)
    scaler = MinMaxScaler()
    data_scaled = list(scaler.fit_transform(data1))
    return data_scaled,y
    
    

if __name__ == "__main__":
    infile = open("champ_data3.json","r")
    data = json.load(infile)
    infile.close()
    cleaned_data = []
    for entry in data: 
        if len(entry)==10:
            cleaned_data.append(entry)
    #dataX,dataY = build_training_set(cleaned_data)
    infile = open("win_data3.json","r")
    y = json.load(infile)
    data1 = []
    dataY = []
    for i in range(len(data)): 
        if len(data[i]) !=16:
            print(len(data[i]))
            print(data[i])
        else:
            data1.append(data[i])
            dataY.append(y[i])
    scaler = MinMaxScaler()
    dataX = scaler.fit_transform(data1)
    print(len(dataX))
    print(len(dataY))
    X = np.asarray(dataX)
    Y = np.asarray(dataY)
    X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.15)
    print(y_test.count(0))
    # Start a run, tracking hyperparameters
    wandb.init(
        # set the wandb project where this run will be logged
        project="325proj",

        # track hyperparameters and run metadata with wandb.config
        config={
        "architecture": "Linear",
        "epochs": 100
        
        }
    )
    config = wandb.config

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X_train,y_train)
    # y_pred = clf.predict(X_test)
    # accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    # print(accuracy)
    # print(sklearn.metrics.confusion_matrix(y_test,y_pred))

    # knn_clf = KNeighborsClassifier(n_neighbors=100)
    # knn_clf.fit(X_train, y_train)
    # y_pred = knn_clf.predict(X_test)
    # accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    # print(accuracy)
    # print(sklearn.metrics.confusion_matrix(y_test,y_pred))

    # forest_clf = RandomForestClassifier()
    # forest_clf.fit(X_train,y_train)
    # y_pred = forest_clf.predict(X_test)
    # accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    # print(accuracy)
    # print(sklearn.metrics.confusion_matrix(y_test,y_pred))

    


    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(target_shape=(16,),input_shape=(16,)),
        tf.keras.layers.Dense(units = 20,activation='tanh'),
        tf.keras.layers.Dense(units = 20,activation='tanh'),
        tf.keras.layers.Dense(units=20,activation='tanh'),
        tf.keras.layers.Dense(units = 20,activation='tanh'),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units = 20,activation='tanh'),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units=2,activation='softmax')
    ])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join("C:/Users/squat/OneDrive/Desktop/CPSC322/CPSC325/CPSC325_Project", 'ckpt', "{epoch:02d}-{val_loss:.4f}.hdf5"),
            monitor = 'val_loss',
            verbose = 0,
            save_best_only = True,
            save_weights_only = False,
            mode = 'auto',
            save_freq='epoch',
            options=None,
            initial_value_threshold=None,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=40,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
        ),
        tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=50,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0
    )
    ]

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    history = model.fit(X_train,y_train,batch_size=50,callbacks = callbacks, epochs=20,validation_data=(X_test,y_test))
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    print(list(y_pred).count(0))
    print(list(y_pred).count(1))
    wandb.finish


