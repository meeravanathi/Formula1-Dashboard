import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load datasets
drivers = pd.read_csv("drivers.csv")
results = pd.read_csv("results.csv")
races = pd.read_csv("races.csv")
lap_times = pd.read_csv("lap_times.csv")
constructor_standings = pd.read_csv("constructor_standings.csv")
driver_standings = pd.read_csv("driver_standings.csv")
sprint_results = pd.read_csv("sprint_results.csv")

# Merge datasets for comprehensive analysis
driver_results = pd.merge(results, drivers, on='driverId')
driver_results = pd.merge(driver_results, races, on='raceId')
data = pd.merge(results, races, on="raceId")
data = pd.merge(data, drivers, on="driverId")
data = pd.merge(data, constructor_standings, on=["constructorId", "raceId"])
sprint_data = pd.merge(sprint_results, drivers, on="driverId")
sprint_data = pd.merge(sprint_data, races, on="raceId")

# Feature Engineering for Predictions
data['race_year'] = data['year']
features = data[['race_year', 'grid', 'laps', 'points_x']]  # Selected features for training
target = data['positionOrder']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Model Training with Gradient Boosting Regressor
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Streamlit Dashboard
st.title("F1 Performance Dashboard")

# Sidebar for Filtering
st.sidebar.title("Filters")
selected_driver = st.sidebar.selectbox("Select Driver", drivers['surname'].unique())
selected_season = st.sidebar.selectbox("Select Season", races['year'].unique())

# Filter data based on selection
filtered_data = driver_results[
    (driver_results['surname'] == selected_driver) & 
    (driver_results['year'] == selected_season)
]
filtered_sprint_data = sprint_data[
    (sprint_data['surname'] == selected_driver) & 
    (sprint_data['year'] == selected_season)
]
filtered_constructor_data = constructor_standings[constructor_standings['raceId'].isin(filtered_data['raceId'])]

# Driver Profile
st.header(f"Driver Profile: {selected_driver}")
driver_info = drivers[drivers['surname'] == selected_driver].iloc[0]
st.write(f"**Full Name**: {driver_info['forename']} {driver_info['surname']}")
st.write(f"**Nationality**: {driver_info['nationality']}")

# Performance Metrics
total_races = filtered_data['raceId'].nunique()
total_wins = filtered_data[filtered_data['positionOrder'] == 1].shape[0]
average_finish = filtered_data['positionOrder'].mean()

st.subheader("Performance Metrics")
st.write(f"**Total Races**: {total_races}")
st.write(f"**Total Wins**: {total_wins}")
st.write(f"**Average Finish**: {average_finish:.2f}")

# Race Results Visualization
fig = px.bar(filtered_data, x='name', y='points', title="Points Per Race")
st.plotly_chart(fig)

# Sprint Race Results
st.subheader("Sprint Race Results")
if not filtered_sprint_data.empty:
    sprint_fig = px.bar(filtered_sprint_data, x='name', y='points', title="Sprint Points Per Race")
    st.plotly_chart(sprint_fig)
else:
    st.write("No sprint race data available for the selected driver and season.")

# Constructor Standings
st.subheader("Constructor Standings")
if not filtered_constructor_data.empty:
    constructor_fig = px.line(filtered_constructor_data, x='raceId', y='points', color='constructorId', 
                              title="Constructor Points Over the Season")
    st.plotly_chart(constructor_fig)
else:
    st.write("No constructor standings data available for the selected season.")

# Lap Time Analysis
if not lap_times.empty:
    lap_data = lap_times[lap_times['driverId'] == driver_info['driverId']]
    lap_data = pd.merge(lap_data, races, on='raceId')
    fig2 = px.line(lap_data, x='lap', y='milliseconds', color='name', title="Lap Times")
    st.plotly_chart(fig2)
else:
    st.write("No lap time data available.")

# Standings Overview
standings_data = driver_standings[
    (driver_standings['driverId'] == driver_info['driverId']) &
    (driver_standings['raceId'].isin(filtered_data['raceId']))
]
if not standings_data.empty:
    fig3 = px.line(standings_data, x='raceId', y='position', title="Driver Standings Over the Season")
    st.plotly_chart(fig3)
else:
    st.write("No standings data available for the selected driver and season.")

# Driver Statistics Over Time
st.header(f"Statistics for {selected_driver}")
driver_data = data[data['surname'] == selected_driver]
if not driver_data.empty:
    fig4 = px.bar(driver_data, x='race_year', y='points_x', title="Points Over Years")
    st.plotly_chart(fig4)
else:
    st.write("No data available for the selected driver.")

# Predict Race Outcome
st.header("Predict Race Outcome")
st.write(f"RMSE of the Gradient Boosting model: {rmse:.2f}")

# Interactive Prediction
user_input = st.slider("Grid Position", 1, 20, 10)
predicted_position = model.predict([[2023, user_input, 60, 25]])[0]
st.write(f"Predicted Position: {int(round(predicted_position))}")
