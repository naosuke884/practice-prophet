# %% [markdown]
# # description
# Follow [the document](https://facebook.github.io/prophet/docs/quick_start.html)

# %%
import pandas as pd
from prophet import Prophet

# %%
# A time series of the log daily page views for the Wikipedia page for [Python Manning](https://en.wikipedia.org/wiki/Peyton_Manning).
df = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)
df.head()
# %%
df.info()
# %%
model = Prophet()

# %%
# fit
model.fit(df)
# %%
# Return only the ds column with historical data and one year of future data.
future = model.make_future_dataframe(periods=365)
future.tail()
# %%
# Predict future.
# TODO: What uncertainly interval ?
# Uncertainty interval: [yhat_lower, yhat_upper].
forecast = model.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

# %%
fig_plot = model.plot(forecast)
# %%
# Plot every component which be trend, weekly or yearly.
fig_plot_components = model.plot_components(forecast)
# %%
# interactive plot with plotly
# need additional installing package
from prophet.plot import plot_plotly

plot_plotly(model, forecast)

# %%
from prophet.plot import plot_components_plotly

plot_components_plotly(model, forecast)
