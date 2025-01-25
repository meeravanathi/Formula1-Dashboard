# Formula1-Dashboard

This project is a comprehensive **Formula 1 Performance Dashboard** built with **Streamlit**. It provides insights into driver statistics, race results, sprint results, constructor standings, and more. It also includes a machine learning model to predict race outcomes.

## Features

### 1. Driver Profile
- Displays driver information such as full name and nationality.
- Performance metrics: total races, total wins, and average finish.

### 2. Race Results
- Visualize driver performance with bar charts showing points scored per race.

### 3. Sprint Race Analysis
- Displays sprint race results and points earned by the selected driver.

### 4. Constructor Standings
- Line chart showing constructor points over the selected season.

### 5. Lap Time Analysis
- Line chart visualizing lap times across races.

### 6. Driver Standings Overview
- Line chart tracking the driver’s standings position over the selected season.

### 7. Machine Learning Predictions
- Predict race outcomes based on grid position, laps, and points using a **Gradient Boosting Regressor**.
- Shows the **Root Mean Squared Error (RMSE)** for model performance.

---

## Installation

### Prerequisites
- Python 3.7+
- Required libraries: `streamlit`, `pandas`, `plotly`, `scikit-learn`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/f1-performance-dashboard.git
   cd f1-performance-dashboard
* **Data Source:** The application utilizes multiple CSV files containing F1 racing data:
    * Driver details (name, nationality, etc.)
    * Race results
    * Lap times
    * Standings (constructor and driver)
    * Sprint results
* **User Interface:**
    * **Sidebar Filters:** Users can filter data by driver and season.
* **Outputs:**
    * **Visualizations:** The app displays visualizations of:
        * Race results
        * Sprint results
        * Constructor standings
        * Lap times
    * **Predictions:** A machine learning model predicts race outcomes based on user-defined grid position.
* **Machine Learning Model:**
    * **Type:** Gradient Boosting Regressor
    * **Features:** Year, grid position, laps, points
    * **Target:** Final position order
    * **Evaluation Metric:** Root Mean Squared Error (RMSE)

