import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prediction import createPredictionModel

TRAINING_DATA_RATIO = 0.8


def prepareDataToDisplay(df, modelName):
    try:
        model = load_model(str(modelName).lower() + "_lstm_model.keras")
        df.index = df.index.tz_convert("Asia/Saigon")
        df["Date"] = df.index

        data = df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])

        for i in range(0, len(data)):
            new_data["Date"][i] = data["Date"][i]
            new_data["Close"][i] = data["Close"][i]

        new_data.index = new_data.Date
        new_data.drop("Date", axis=1, inplace=True)

        dataset = new_data.values

        separation_index = int(TRAINING_DATA_RATIO * len(dataset))

        train = dataset[0:separation_index, :]
        valid = dataset[separation_index:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []

        for i in range(60, len(train)):
            x_train.append(scaled_data[i - 60 : i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        inputs = new_data[len(new_data) - len(valid) - 60 :].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i - 60 : i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        valid = new_data[separation_index:]
        valid["Predictions"] = closing_price

        return valid
    except OSError:
        createPredictionModel(df, modelName)
        return prepareDataToDisplay(df, modelName)
