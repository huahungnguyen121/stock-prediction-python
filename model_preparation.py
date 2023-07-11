# Data Source
import yfinance as yf

from prediction import createPredictionModel

# Arguments to fetch data
tickerTypes = ["BTC-USD", "ETH-USD", "ADA-USD"]
period = "48h"
interval = "1m"


def fetchTrainingData(tickerType):
    df = yf.download(tickers=tickerType, period=period, interval=interval)

    return df


def fetchAllTrainingData():
    dataframes = []
    for ticker in tickerTypes:
        df = fetchTrainingData(ticker)
        dataframes.append(df)

    return dataframes


def prepareTrainingModel(tickerType):
    # Fetch data from the Internet
    dataframe = fetchTrainingData(tickerType)

    # Create model file
    createPredictionModel(dataframe, tickerType.replace("-", "_"))

    print("\nCompleted creating model file for: " + tickerType + "\n")


def prepareAllTrainingModel():
    # Fetch data from the Internet
    dataframes = fetchAllTrainingData()

    # Create model files
    for index, ticker in enumerate(tickerTypes):
        createPredictionModel(dataframes[index], ticker.replace("-", "_"))

    print("\nCompleted creating all model files\n")


if __name__ == "__main__":
    prepareAllTrainingModel()
