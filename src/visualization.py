
import pandas as pd
import plotly.express as px

from data_loader import load_cyclone_data

df = load_cyclone_data()

# Select the columns needed for visualization
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")

# Remove rows without usable time or wind values
plot_df = df.dropna(subset=["ISO_TIME", "USA_WIND"]).copy()

# Keep the strongest observation for each year
plot_df["YEAR"] = plot_df["ISO_TIME"].dt.year

yearly_wind = (
    plot_df.groupby("YEAR", as_index=False)["USA_WIND"]
    .max()
    .rename(columns={"USA_WIND": "MAX_WIND"})
)

fig = px.line(
    yearly_wind,
    x="YEAR",
    y="MAX_WIND",
    markers=True,
    title="Maximum Recorded Cyclone Wind by Year",
    labels={
        "YEAR": "Year",
        "MAX_WIND": "Wind Speed (knots)"
    },
    template="plotly_dark"
)

fig.update_layout(
    title_x=0.5,
    xaxis_title="Year",
    yaxis_title="Wind Speed (knots)",
    hovermode="x unified"
)

fig.show()
