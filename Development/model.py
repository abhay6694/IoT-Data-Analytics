from flask import Flask
import paho.mqtt
import pandas as pd
import paho.mqtt.subscribe as subscribe
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics import tsaplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import sys

app = Flask(__name__)
fileName = 'enviornment_data.json'
split_day = '2018-10-13'
model_path = 'pipeline_model.pkl'
model_accuracy_threshold = 0.90
overloaded_cached_data_file = 'data.txt'
topics = []
hostname = '0.0.0.0'
cache = []


@app.route("/")
def deviceMonitoring():
    ## Uncomment this code if want to process stream data
    # print('Stream Data is saving........')
    # subscribe.callback(cacheStreamData, topics=topics, hostname=hostname)
    #
    # print('Stream Data is processing........')
    # subscribe.callback(processStreamData, topics=topics, hostname=hostname)

    print('Data is loading ......')
    sampledDF = getSampledData(fileName)

    # exploreData(sampledDF)
    print('Data is splitting .......')
    X_train, y_train, X_test, y_test = splitData(sampledDF, split_day)
    print('Model is saving.......')
    model = saveModel(X_train, y_train)
    model_current_accuracy = getAccuracyScore(X_test, y_test)

    status = ''
    if model_current_accuracy < model_accuracy_threshold:
        status = 'Accuracy is not up-to-date. Please update the model !!!!!!!!!!!!!!'

    # Use stream data if data is continously streaming.
    predictions = model.predict(X_test)
    status = 'Model is working Fine'

    return status, predictions


def cacheStreamData(client, userdata, message):
    data = json.loads(message.payload)
    cache.append(data)
    cache_status = 0
    if len(cache) > MAX_CACHE:
        with Path(overloaded_cached_data_file).open("a") as f:
            f.writelines(cache)
        cache.clear()
        cache_status = 1
    # Return a 1 as status when cache is full otherwise 0.
    return cache_status


def processStreamData(client, userdata, message, model):
    data = json.loads(message.payload)
    df = pd.DataFrame.from_records([data], index="timestamp", columns=cols)
    return model.predict(df)


def getSampledData(file_Name):
    env = pd.read_json(file_Name)
    env["timestamp"] = pd.to_datetime(env["timestamp"], unit="ms")
    env.set_index('timestamp', inplace=True)
    df_res = env.resample("10min").last()
    return df_res


def exploreData(sampledDF):
    sampledDF[["temperature", "pressure"]].plot(title="Environment", secondary_y="pressure")
    plt.ylabel('Temperature')
    plt.ylabel('Pressure')
    plt.show()

    sampledDF[['humidity', 'pressure', 'radiation', 'temperature']].hist(bins=20)
    plt.show()

    sampledDF[['humidity', 'temperature']].plot()

    sns.heatmap(sampledDF.corr(), annot=True)
    sns.pairplot(sampledDF)


def splitData(sampledDF, split_day):
    train = sampledDF[:split_day]
    test = sampledDF[split_day:]

    X_train = train.drop("target", axis=1)
    y_train = train["target"]
    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    return X_train, y_train, X_test, y_test


def saveModel(X_train, y_train):
    # Initialize Objects
    sc = StandardScaler()
    logreg = LogisticRegression()
    # Create pipeline
    pl = Pipeline([
        ("scale", sc),
        ("logreg", logreg)
    ])

    pl.fit(X_train, y_train)

    with Path(model_path).open("bw") as f:
        pickle.dump(pl, f)

    return pl


def getAccuracyScore(X_test, y_test):
    with Path(model_path).open('br') as f:
        pl = pickle.load(f)

    return pl.score(X_test, y_test)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
