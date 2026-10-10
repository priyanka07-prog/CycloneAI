import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from src.data_loader import load_cyclone_data


REQUIRED_COLUMNS = {
    "SID",
    "SEASON",
    "BASIN",
    "NAME",
    "LAT",
    "LON",
    "USA_WIND",
    "WMO_WIND",
}


def prepare_storm_summary(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required IBTrACS columns: "
            + ", ".join(sorted(missing_columns))
        )

    observations = data[
        ["SID", "SEASON", "BASIN", "NAME", "LAT", "LON", "USA_WIND", "WMO_WIND"]
    ].copy()
    for column in ("SEASON", "LAT", "LON", "USA_WIND", "WMO_WIND"):
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
    observations["MAX_WIND_KT"] = observations["USA_WIND"].combine_first(
        observations["WMO_WIND"]
    )

    located_observations = observations.dropna(subset=["SID", "SEASON", "LAT", "LON"])
    wind_observations = located_observations.dropna(subset=["MAX_WIND_KT"])
    peak_indices = wind_observations.groupby("SID")["MAX_WIND_KT"].idxmax()
    storms_with_wind = wind_observations.loc[peak_indices]
    storms_without_wind = located_observations[
        ~located_observations["SID"].isin(storms_with_wind["SID"])
    ].drop_duplicates("SID", keep="first")
    return pd.concat([storms_with_wind, storms_without_wind]).reset_index(drop=True)


DATA = load_cyclone_data()
STORMS = prepare_storm_summary(DATA)
MIN_YEAR = int(STORMS["SEASON"].min())
MAX_YEAR = int(STORMS["SEASON"].max())

app = Dash(__name__)
app.title = "CycloneAI | IBTrACS Explorer"

PAGE_STYLE = {
    "backgroundColor": "#0b1220",
    "color": "#e5e7eb",
    "fontFamily": "Arial, sans-serif",
    "minHeight": "100vh",
    "padding": "28px",
}
PANEL_STYLE = {
    "backgroundColor": "#111c2f",
    "border": "1px solid #26354c",
    "borderRadius": "12px",
    "padding": "18px",
}
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

app.layout = html.Main(
    style=PAGE_STYLE,
    children=[
        html.Header(
            [
                html.Div(
                    "IBTrACS DATA EXPLORER",
                    style={
                        "color": "#38bdf8",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                        "letterSpacing": "2px",
                    },
                ),
                html.H1("Tropical Cyclone Tracker", style={"margin": "8px 0"}),
                html.P(
                    "Explore storm intensity, locations, seasons, and ocean basins.",
                    style={"color": "#a7b4c8"},
                ),
            ],
            style={"marginBottom": "24px"},
        ),
        html.Section(
            [
                html.Label("Season range", htmlFor="season-range"),
                dcc.RangeSlider(
                    id="season-range",
                    min=MIN_YEAR,
                    max=MAX_YEAR,
                    value=[MIN_YEAR, MAX_YEAR],
                    marks={year: str(year) for year in range(MIN_YEAR, MAX_YEAR + 1, 5)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": False},
                    allowCross=False,
                ),
            ],
            style={**PANEL_STYLE, "marginBottom": "18px", "padding": "20px 24px 30px"},
        ),
        html.Section(
            [
                html.Div(
                    [html.Div("STORMS", className="metric-label"), html.H2(id="storm-count")],
                    style=PANEL_STYLE,
                ),
                html.Div(
                    [
                        html.Div("STRONGEST WIND", className="metric-label"),
                        html.H2(id="strongest-wind"),
                    ],
                    style=PANEL_STYLE,
                ),
                html.Div(
                    [
                        html.Div("STRONGEST STORM", className="metric-label"),
                        html.H2(id="strongest-name"),
                    ],
                    style=PANEL_STYLE,
                ),
                html.Div(
                    [
                        html.Div("YEARS SELECTED", className="metric-label"),
                        html.H2(id="years-selected"),
                    ],
                    style=PANEL_STYLE,
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "14px",
                "marginBottom": "18px",
            },
        ),
        html.Section(
            [
                html.Div(
                    [
                        html.H3("Peak intensity locations"),
                        dcc.Graph(id="storm-map", config=GRAPH_CONFIG),
                    ],
                    style=PANEL_STYLE,
                ),
                html.Div(
                    [
                        html.H3("Storms by season"),
                        dcc.Graph(id="season-chart", config=GRAPH_CONFIG),
                    ],
                    style=PANEL_STYLE,
                ),
                html.Div(
                    [
                        html.H3("Storms by basin"),
                        dcc.Graph(id="basin-chart", config=GRAPH_CONFIG),
                    ],
                    style={**PANEL_STYLE, "gridColumn": "1 / -1"},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                "gap": "14px",
            },
        ),
        html.Footer(
            "Source: NOAA/NCEI IBTrACS. Wind values are knots; map points show each storm's strongest recorded observation.",
            style={"color": "#94a3b8", "fontSize": "12px", "padding": "18px 2px"},
        ),
    ],
)


@app.callback(
    Output("storm-count", "children"),
    Output("strongest-wind", "children"),
    Output("strongest-name", "children"),
    Output("years-selected", "children"),
    Output("storm-map", "figure"),
    Output("season-chart", "figure"),
    Output("basin-chart", "figure"),
    Input("season-range", "value"),
)
def update_dashboard(year_range: list[int]) -> tuple[object, ...]:
    start_year, end_year = year_range
    filtered = STORMS[STORMS["SEASON"].between(start_year, end_year)]
    mapped_storms = filtered.dropna(subset=["MAX_WIND_KT"])

    if mapped_storms.empty:
        strongest_wind = "N/A"
        strongest_name = "No storms"
        map_figure = px.scatter_geo(template="plotly_dark")
    else:
        peak_storm = mapped_storms.loc[mapped_storms["MAX_WIND_KT"].idxmax()]
        strongest_wind = f"{peak_storm['MAX_WIND_KT']:.0f} kt"
        strongest_name = str(peak_storm["NAME"])
        map_figure = px.scatter_geo(
            mapped_storms,
            lat="LAT",
            lon="LON",
            color="BASIN",
            size="MAX_WIND_KT",
            size_max=16,
            hover_name="NAME",
            hover_data={
                "SEASON": True,
                "MAX_WIND_KT": ":.0f",
                "LAT": ":.1f",
                "LON": ":.1f",
            },
            projection="natural earth",
            template="plotly_dark",
            labels={"MAX_WIND_KT": "Peak wind (kt)", "BASIN": "Basin"},
        )
    map_figure.update_layout(
        height=510,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend_title_text="Basin",
    )
    storm_count = f"{filtered['SID'].nunique():,}"
    if filtered.empty:
        storm_count = "0"
        strongest_name = "No storms"

    season_counts = (
        filtered.groupby("SEASON")["SID"]
        .nunique()
        .rename("Storms")
        .reset_index()
    )
    season_figure = px.line(
        season_counts,
        x="SEASON",
        y="Storms",
        markers=True,
        template="plotly_dark",
        labels={"SEASON": "Season", "Storms": "Storms"},
    )
    season_figure.update_layout(height=350, margin={"l": 10, "r": 10, "t": 10, "b": 10})

    basin_counts = (
        filtered.groupby("BASIN")["SID"]
        .nunique()
        .sort_values(ascending=True)
        .rename("Storms")
        .reset_index()
    )
    basin_figure = px.bar(
        basin_counts,
        x="BASIN",
        y="Storms",
        color="BASIN",
        template="plotly_dark",
        labels={"BASIN": "Basin", "Storms": "Storms"},
    )
    basin_figure.update_layout(
        height=320,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )

    return (
        storm_count,
        strongest_wind,
        strongest_name,
        f"{start_year}–{end_year}",
        map_figure,
        season_figure,
        basin_figure,
    )


if __name__ == "__main__":
    app.run(debug=False)