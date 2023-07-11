import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from utils import prepareDataToDisplay
from model_preparation import fetchTrainingData, tickerTypes

app = dash.Dash()
server = app.server


def prepare(tickerType):
    dataframe = fetchTrainingData(tickerType)

    return prepareDataToDisplay(dataframe, tickerType.replace("-", "_"))


def main():
    app.layout = html.Div(
        [
            dcc.Store(id="loading-state", data=False),
            html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
            dcc.Loading(
                id="loading-indicator",
                fullscreen=True,
            ),
            html.Div(
                [
                    html.H2(
                        "Choose a pair of currencies",
                        style={"textAlign": "center"},
                    ),
                    dcc.Dropdown(
                        id="choose-coin-dropdown",
                        options=[
                            {
                                "label": tickerTypes[0],
                                "value": 0,
                            },
                            {
                                "label": tickerTypes[1],
                                "value": 1,
                            },
                            {
                                "label": tickerTypes[2],
                                "value": 2,
                            },
                        ],
                        value=0,
                        style={
                            "display": "block",
                            "margin-left": "auto",
                            "margin-right": "auto",
                            "width": "60%",
                        },
                    ),
                    html.H2(
                        "Actual closing price",
                        style={"textAlign": "center"},
                    ),
                    dcc.Graph(
                        id="actual-data",
                        figure={
                            "layout": go.Layout(
                                title="N/A",
                                xaxis={"title": "Date"},
                                yaxis={"title": "Price"},
                            ),
                        },
                    ),
                    html.H2(
                        "LSTM Predicted closing price",
                        style={"textAlign": "center"},
                    ),
                    dcc.Graph(
                        id="predicted-data",
                        figure={
                            "layout": go.Layout(
                                title="N/A",
                                xaxis={"title": "Date"},
                                yaxis={"title": "Price"},
                            ),
                        },
                    ),
                    html.H2(
                        "Compare actual and predicted closing prices",
                        style={"textAlign": "center"},
                    ),
                    dcc.Graph(
                        id="comparation-graph",
                        figure={
                            "layout": go.Layout(
                                title="N/A",
                                xaxis={"title": "Date"},
                                yaxis={"title": "Price"},
                            ),
                        },
                    ),
                ]
            ),
        ]
    )

    @app.callback(
        [
            Output("actual-data", "figure"),
            Output("predicted-data", "figure"),
            Output("comparation-graph", "figure"),
            Output("loading-indicator", "children"),
        ],
        Input("choose-coin-dropdown", "value"),
    )
    def update_prediction_graph(selected_dropdown):
        data = prepare(tickerTypes[selected_dropdown])
        figure1 = {
            "data": [
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="markers",
                )
            ],
            "layout": go.Layout(
                title=tickerTypes[selected_dropdown],
                xaxis={"title": "Date"},
                yaxis={"title": "Price"},
            ),
        }

        figure2 = {
            "data": [
                go.Scatter(
                    x=data.index,
                    y=data["Predictions"],
                    mode="markers",
                )
            ],
            "layout": go.Layout(
                title=tickerTypes[selected_dropdown],
                xaxis={"title": "Date"},
                yaxis={"title": "Price"},
            ),
        }

        trace1 = []
        trace2 = []
        trace1.append(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                opacity=0.7,
                name=f"Actual {tickerTypes[selected_dropdown]}",
                textposition="bottom center",
            )
        )
        trace2.append(
            go.Scatter(
                x=data.index,
                y=data["Predictions"],
                mode="lines",
                opacity=0.6,
                name=f"Predicted {tickerTypes[selected_dropdown]}",
                textposition="bottom center",
            )
        )
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]

        figure3 = {
            "data": data,
            "layout": go.Layout(
                colorway=[
                    "#5E0DAC",
                    "#FF4F00",
                    "#375CB1",
                    "#FF7400",
                    "#FFF400",
                    "#FF0056",
                ],
                height=600,
                title=f"Actual and predicted closing prices for {str(tickerTypes[selected_dropdown])} over time",
                xaxis={"title": "Date"},
                yaxis={"title": "Price"},
            ),
        }

        return figure1, figure2, figure3, None


if __name__ == "__main__":
    main()
    app.run(debug=True, use_reloader=False)
