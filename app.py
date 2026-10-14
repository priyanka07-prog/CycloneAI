from pathlib import Path

import joblib
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.data_loader import load_cyclone_data


def style_cyclone_chart(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080b14",
        plot_bgcolor="#101525",
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#dbe4f0",
            size=12,
        ),
        title=dict(
            text=title if title else "",
            font=dict(size=20, color="#c084fc"),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=25, r=25, t=65, b=25),
        hoverlabel=dict(
            bgcolor="#151d32",
            bordercolor="#a855f7",
            font=dict(color="#ffffff", size=13),
        ),
        legend=dict(
            bgcolor="rgba(16, 21, 37, 0.75)",
            bordercolor="#29334d",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        zeroline=False,
    )

    return fig

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "artifacts" / "cyclone_intensity_model.joblib"
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]

df = load_cyclone_data()

df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
for column in ("LAT", "LON", "USA_WIND"):
    df[column] = pd.to_numeric(df[column], errors="coerce").replace(-999, float("nan"))

plot_df = df.dropna(subset=["ISO_TIME", "LAT", "LON", "USA_WIND"]).copy()
plot_df["YEAR"] = plot_df["ISO_TIME"].dt.year

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

years = sorted(plot_df["YEAR"].dropna().unique())

app.layout = dbc.Container(
    [
        html.H1("CycloneAI | Storm Analytics", className="text-info text-center mt-4"),
        html.H2("Model Performance", className="text-info mt-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Accuracy"),
                                html.H3("86.4%", className="text-info"),
                                html.P("Overall correct predictions"),
                            ]
                        ),
                        color="dark",
                        inverse=True,
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Macro F1-score"),
                                html.H3("0.87", className="text-info"),
                                html.P("Balanced performance across classes"),
                            ]
                        ),
                        color="dark",
                        inverse=True,
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Test observations"),
                                html.H3("24,639", className="text-info"),
                                html.P("Held-out observations evaluated"),
                            ]
                        ),
                        color="dark",
                        inverse=True,
                    ),
                    md=4,
                ),
            ],
            className="g-3 mb-4",
        ),
        html.H2("Cyclone Intensity Prediction", className="text-info mt-4 mb-3"),
        html.Label("Latitude"),
        dcc.Input(
            id="input-lat",
            type="number",
            min=-90,
            max=90,
            value=15,
            step=0.1,
        ),
        html.Label("Longitude"),
        dcc.Input(
            id="input-lon",
            type="number",
            min=-180,
            max=180,
            value=75,
            step=0.1,
        ),
        html.Label("Month (1-12)"),
        dcc.Input(
            id="input-month",
            type="number",
            min=1,
            max=12,
            value=7,
            step=1,
        ),
        html.Label("Pressure (hPa)"),
        dcc.Input(
            id="input-pressure",
            type="number",
            min=800,
            max=1100,
            value=1000,
            step=1,
        ),
        html.Button("Predict Intensity", id="predict-button", n_clicks=0),
        html.Div(id="prediction-result"),
        html.P(
            "Historical Tropical Cyclone Data Explorer",
            className="text-center text-light",
        ),
        dbc.Label("Select Year", className="text-info mt-3"),
        dcc.Dropdown(
            id="year-filter",
            options=[{"label": "All Years", "value": "all"}] + [
                {"label": str(int(year)), "value": int(year)}
                for year in years
            ],
            value="all",
            clearable=False,
            className="mb-4",
        ),
        html.H2("Global Cyclone Tracking", className="text-info mt-4 mb-3"),
        dcc.Graph(id="wind-chart"),
    ],
    fluid=True,
    className="p-4",
)

@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    State("input-lat", "value"),
    State("input-lon", "value"),
    State("input-month", "value"),
    State("input-pressure", "value"),
)
def predict_intensity(n_clicks, lat, lon, month, pressure):
    if not n_clicks:
        return "Enter cyclone conditions and click Predict Intensity."

    if None in (lat, lon, month, pressure):
        return "Please fill in all four inputs."

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "Latitude or longitude is outside the valid range."

    if not (1 <= month <= 12):
        return "Month must be between 1 and 12."

    if not float(month).is_integer():
        return "Month must be a whole number between 1 and 12."

    if not (800 <= pressure <= 1100):
        return "Enter a pressure between 800 and 1100 hPa."

    input_data = pd.DataFrame(
        [[lat, lon, month, pressure]],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    return f"Predicted cyclone intensity: {prediction}"


@app.callback(
    Output("wind-chart", "figure"),
    Input("year-filter", "value"),
)
def update_chart(selected_year):
    filtered = plot_df.copy()

    if selected_year != "all":
        filtered = filtered[filtered["YEAR"] == int(selected_year)]

    if selected_year == "all" and len(filtered) > 5000:
        filtered = filtered.sample(n=5000, random_state=42)

    fig = px.scatter_geo(
        filtered,
        lat="LAT",
        lon="LON",
        color="USA_WIND",
        hover_name="NAME",
        hover_data={
            "ISO_TIME": True,
            "USA_WIND": True,
            "LAT": True,
            "LON": True,
        },
        color_continuous_scale="Turbo",
        projection="natural earth",
        title="Historical Cyclone Observation Locations",
        labels={
            "USA_WIND": "Wind speed (knots)",
            "ISO_TIME": "Observation time",
            "LAT": "Latitude",
            "LON": "Longitude",
        },
        template="plotly_dark",
    )

    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.8,
            line=dict(width=0.4, color="#e2e8f0"),
        ),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Wind: %{marker.color:.1f} knots<br>"
            "Latitude: %{lat:.2f}<br>"
            "Longitude: %{lon:.2f}"
            "<extra></extra>"
        ),
    )

    fig = style_cyclone_chart(
        fig,
        title="Global Historical Cyclone Observations",
    )

    fig.update_layout(
        height=580,
        title_x=0.5,
        geo=dict(
            showland=True,
            landcolor="#172536",
            showocean=True,
            oceancolor="#081522",
            showcountries=True,
            countrycolor="#334155",
            showcoastlines=True,
            coastlinecolor="#475569",
            showframe=False,
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title="Wind<br>(knots)",
            tickfont=dict(color="#dbe4f0"),
            title_font=dict(color="#dbe4f0"),
            bgcolor="#101525",
            outlinewidth=0,
        ),
    )

    fig.update_traces(
        marker=dict(
            size=6,
            opacity=0.9,
            line=dict(width=0.5, color="#e2e8f0"),
        ),
        selector=dict(type="scattergeo"),
    )

    fig.update_layout(
        height=650,
        paper_bgcolor="#080b14",
        font=dict(
            family="Arial, sans-serif",
            color="#e2e8f0",
            size=13,
        ),
        margin=dict(l=10, r=20, t=75, b=15),
        title=dict(
            text="GLOBAL CYCLONE OBSERVATIONS",
            x=0.03,
            xanchor="left",
            font=dict(size=21, color="#c084fc"),
        ),
        coloraxis=dict(
            cmin=0,
            cmax=150,
            colorscale=[
                [0.0, "#38bdf8"],
                [0.35, "#22c55e"],
                [0.65, "#facc15"],
                [1.0, "#f43f5e"],
            ],
            colorbar=dict(
                title=dict(text="Wind (knots)", font=dict(color="#e2e8f0")),
                tickfont=dict(color="#e2e8f0"),
                bgcolor="#101525",
                outlinewidth=0,
                thickness=15,
                len=0.75,
            ),
        ),
        geo=dict(
            projection=dict(type="natural earth", scale=1.05),
            bgcolor="#080b14",
            showland=True,
            landcolor="#26364b",
            showocean=True,
            oceancolor="#07111f",
            showcountries=True,
            countrycolor="#52647a",
            showcoastlines=True,
            coastlinecolor="#8ba1b8",
            coastlinewidth=0.7,
            showframe=False,
            showlakes=True,
            lakecolor="#07111f",
        ),
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
