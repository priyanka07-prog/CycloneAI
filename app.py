import plotly.express as px
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from src.data_loader import load_cyclone_data



from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent

# Locate the saved machine-learning model
MODEL_PATH = PROJECT_ROOT / "artifacts" / "cyclone_intensity_model.joblib"

# Load the trained model bundle
model_bundle = joblib.load(MODEL_PATH)

# Get the trained pipeline and its feature information
model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]

print("CycloneAI model loaded successfully!")
print("Model features:", feature_columns)

# Load historical data
df = load_cyclone_data()

df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")

map_df = df.dropna(
    subset=["LAT", "LON", "ISO_TIME"]
).copy()

map_df["YEAR"] = map_df["ISO_TIME"].dt.year

# Prepare data for the chart
plot_df = df.dropna(subset=["ISO_TIME", "USA_WIND"]).copy()
plot_df["YEAR"] = plot_df["ISO_TIME"].dt.year

yearly = (
    plot_df.groupby("YEAR", as_index=False)["USA_WIND"]
    .max()
    .rename(columns={"USA_WIND": "MAX_WIND"})
)

html.H2(
    "Global Cyclone Tracking",
    className="text-info mt-4 mb-3"
),

dcc.Graph(id="cyclone-map"),


# Create the dashboard
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG]
)

years = sorted(plot_df["YEAR"].dropna().unique())

app.layout = dbc.Container(
    [
        html.H1(
            "CycloneAI | Storm Analytics",
            className="text-info text-center mt-4"
        ),
        
     # Cyclone intensity prediction form
      html.H2("Cyclone Intensity Prediction"),

        html.Label("Latitude"),
        dcc.Input(
        id="input-lat",
        type="number",
        min=-90,
        max=90,
        value=15,
        step=0.1
    ),

    html.Label("Longitude"),
    dcc.Input(
    id="input-lon",
    type="number",
    min=-180,
    max=180,
    value=75,
    step=0.1
),

html.Label("Month (1-12)"),
dcc.Input(
    id="input-month",
    type="number",
    min=1,
    max=12,
    value=7,
    step=1
    ),

   html.Label("Pressure (hPa)"),
    dcc.Input(
    id="input-pressure",
    type="number",
    min=800,
    max=1100,
    value=1000,
    step=1
    ),

    html.Button("Predict Intensity", id="predict-button", n_clicks=0),

    html.Div(id="prediction-result"),

        html.P(
            "Historical Tropical Cyclone Data Explorer",
            className="text-center text-light"
        ),
        
         dbc.Label("Select Year", className="text-info mt-3"),
        dcc.Dropdown(
            id="year-filter",
            options=[
                {"label": "All Years", "value": "all"}
            ] + [
                {"label": str(int(year)), "value": int(year)}
                for year in years
            ],
            value="all",
            clearable=False,
            className="mb-4"
        ),
        
        dcc.Graph(id="wind-chart"),
    ],
    fluid=True,
    className="p-4"
)

@app.callback(
    Output("wind-chart", "figure"),
    Input("year-filter", "value")
)
def update_chart(selected_year):
    filtered = plot_df.copy()

    if selected_year != "all":
        filtered = filtered[
            filtered["YEAR"] == selected_year
        ]

# Limit points when all years are selected
    if selected_year == "all" and len(filtered) > 5000:
        filtered = filtered.sample(
            n=5000,
            random_state=42
        )
        
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
            "LON": True
        },
        color_continuous_scale="Turbo",
        projection="natural earth",
        title="Historical Cyclone Observation Locations",
        labels={"USA_WIND": "Wind (knots)"},
        template="plotly_dark"
    )

    fig.update_layout(
        title_x=0.5,
        geo=dict(
            showland=True,
            landcolor="#172536",
            showocean=True,
            oceancolor="#081522"
        )
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
