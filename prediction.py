import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

TRAINING_DATA_RATIO = 0.8


# Take data from the "df" to train, create and save the training model to a file named: modelName_lstm_model.h5
def createPredictionModel(df, modelName):
    df["Date"] = df.index

    data = df.sort_index(ascending=True, axis=0)

    # Prepare training data to create a model
    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])

    for i in range(0, len(data)):
        new_dataset["Date"][i] = data["Date"][i]
        new_dataset["Close"][i] = data["Close"][i]

    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)

    final_dataset = new_dataset.values

    separation_index = int(TRAINING_DATA_RATIO * len(final_dataset))

    train_data = final_dataset[0:separation_index, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60 : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )

    lstm_model = Sequential()
    lstm_model.add(
        LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1))
    )
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

    # save training model to a file
    lstm_model.save(str(modelName).lower() + "_lstm_model.keras")
