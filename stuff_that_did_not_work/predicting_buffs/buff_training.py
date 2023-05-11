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
import pandas as pd


if __name__ == "__main__":
    infile = open("buff_x_train.json","r")
    X = json.load(infile)
    infile.close()
    infile = open("buff_y_train.json","r")
    Y = json.load(infile)
    infile.close()
    X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.15,stratify=Y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    print(accuracy)
    print(sklearn.metrics.confusion_matrix(y_test,y_pred))

    knn_clf = KNeighborsClassifier(n_neighbors=9)
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    print(accuracy)
    print(sklearn.metrics.confusion_matrix(y_test,y_pred))

    forest_clf = RandomForestClassifier(max_depth=5)
    forest_clf.fit(X_train,y_train)
    y_pred = forest_clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    print(accuracy)
    print(sklearn.metrics.confusion_matrix(y_test,y_pred))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(target_shape=(6,),input_shape=(6,)),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units=20,activation='sigmoid'),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units = 20,activation='sigmoid'),
        tf.keras.layers.Dense(units=3,activation='softmax')
    ])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join('ckpt', "{epoch:02d}-{val_loss:.4f}.hdf5"),
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
        )
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    history = model.fit(X_train,y_train,batch_size=50,callbacks = callbacks, epochs=10,validation_data=(X_test,y_test))
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    #print(y_pred)
    print(y_train.count(0)/len(y_train))
    print(y_train.count(1)/len(y_train))
    print(y_train.count(2)/len(y_train))
    accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    print(accuracy)
    print(sklearn.metrics.confusion_matrix(y_test,y_pred))
