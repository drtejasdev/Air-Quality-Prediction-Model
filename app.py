
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sklearn

# Load and prepare the data
df = pd.read_csv('cleaned_aqi_data.csv') # Modified file path
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.rename(columns={'AQI': 'y'})
df = df.rename_axis('ds')

# Handle outliers using IQR
Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[(df['y'] >= lower_bound) & (df['y'] <= upper_bound)].copy() # Use .copy() to avoid SettingWithCopyWarning
df_cleaned_reset = df_cleaned.reset_index().copy() # Use .copy()

# Create a copy of the DataFrame with original 'y' values before log transformation for visualizations
df_original_y = df_cleaned_reset.copy()

# Apply log transformation to 'y' for training
df_cleaned_reset['y'] = df_cleaned_reset['y'].apply(lambda x: np.log1p(x) if x > 0 else np.nan)
df_cleaned_reset.dropna(subset=['y'], inplace=True) # Drop rows with NaN values after transformation


# Train the Prophet model on log-transformed data
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df_cleaned_reset)

st.title('Air Quality Prediction with Prophet (Log Transformed)')

st.write("Select a year and month to predict the air quality.")

# Add sidebar inputs for year and month
year_input = st.selectbox(
    'Select Year for Prediction',
    range(2024, 2031) # Predicting a bit further out
)

month_input = st.selectbox(
    'Select Month for Prediction (1-12)',
    range(1, 13)
)

# Generate future dates for prediction for the selected month and year
future_dates_month = pd.date_range(start=f'{year_input}-{month_input}-01', end=f'{year_input}-{month_input}-{pd.Period(f"{year_input}-{month_input}").days_in_month}', freq='D')
future_month = pd.DataFrame({'ds': future_dates_month})

# Generate predictions
future_forecast_month = model.predict(future_month)

# Inverse transform the predictions
future_forecast_month['yhat'] = np.expm1(future_forecast_month['yhat'])
future_forecast_month['yhat_lower'] = np.expm1(future_forecast_month['yhat_lower'])
future_forecast_month['yhat_upper'] = np.expm1(future_forecast_month['yhat_upper'])


st.subheader(f'Predicted AQI for {pd.to_datetime(f"{year_input}-{month_input}-01").strftime("%B %Y")}')
st.write(future_forecast_month[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Predicted AQI', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}))

# Display prediction interval information
mean_interval_width = (future_forecast_month['yhat_upper'] - future_forecast_month['yhat_lower']).mean() / 2
st.write(f"The prediction for the selected month can be off by approximately ±{mean_interval_width:.2f} AQI.")


# --- Historical vs Predicted AQI for a Selected Year ---
st.subheader(f'Historical vs Predicted AQI for {year_input}')

# Prepare data for the selected year's comparison plot (if historical data exists for that year)
# Need to inverse transform historical predictions as well for comparison
if year_input in df_original_y['ds'].dt.year.unique():
    future_year = model.make_future_dataframe(periods=0, include_history=True)
    forecast_year = model.predict(future_year)
    forecast_year['yhat'] = np.expm1(forecast_year['yhat']) # Inverse transform historical predictions

    # Merge with original historical data (not log-transformed 'y')
    merged_df_year = pd.merge(df_original_y[['ds', 'y']].rename(columns={'y': 'Actual AQI'}),
                              forecast_year[['ds', 'yhat']], on='ds')

    df_year = merged_df_year[merged_df_year['ds'].dt.year == year_input]

    fig_year, ax_year = plt.subplots(figsize=(12, 6))
    ax_year.plot(df_year['ds'], df_year['Actual AQI'], label='Actual AQI')
    ax_year.plot(df_year['ds'], df_year['yhat'], label='Predicted AQI')
    ax_year.set_xlabel('Date')
    ax_year.set_ylabel('AQI')
    ax_year.set_title(f'Actual vs Predicted AQI for {year_input}')
    ax_year.legend()
    st.pyplot(fig_year)
else:
    st.write(f'No historical data available for {year_input}. Showing predicted trend for the year.')
    # If predicting for a future year, show the predicted trend for the whole year
    future_dates_year = pd.date_range(start=f'{year_input}-01-01', end=f'{year_input}-12-31', freq='D')
    future_year_pred = pd.DataFrame({'ds': future_dates_year})
    forecast_year_pred = model.predict(future_year_pred)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] # Include prediction intervals
    # Inverse transform future year predictions
    forecast_year_pred['yhat'] = np.expm1(forecast_year_pred['yhat'])
    forecast_year_pred['yhat_lower'] = np.expm1(forecast_year_pred['yhat_lower'])
    forecast_year_pred['yhat_upper'] = np.expm1(forecast_year_pred['yhat_upper'])


    fig_year_pred, ax_year_pred = plt.subplots(figsize=(12, 6))
    ax_year_pred.plot(forecast_year_pred['ds'], forecast_year_pred['yhat'], label='Predicted AQI')
    ax_year_pred.fill_between(forecast_year_pred['ds'], forecast_year_pred['yhat_lower'], forecast_year_pred['yhat_upper'], color='k', alpha=.2, label='Prediction Interval')
    ax_year_pred.set_xlabel('Date')
    ax_year_pred.set_ylabel('AQI')
    ax_year_pred.set_title(f'Predicted AQI for {year_input}')
    ax_year_pred.legend()
    st.pyplot(fig_year_pred)


st.subheader(f'Predicted AQI for {pd.to_datetime(f"{year_input}-{month_input}-01").strftime("%B %Y")}')

fig_month, ax_month = plt.subplots(figsize=(12, 6))
ax_month.plot(future_forecast_month['ds'], future_forecast_month['yhat'], label='Predicted AQI')
ax_month.fill_between(future_forecast_month['ds'], future_forecast_month['yhat_lower'], future_forecast_month['yhat_upper'], color='k', alpha=.2, label='Prediction Interval')
ax_month.set_xlabel('Date')
ax_month.set_ylabel('AQI')
ax_month.set_title(f'Predicted AQI for {pd.to_datetime(f"{year_input}-{month_input}-01").strftime("%B %Y")}')
ax_month.legend()
st.pyplot(fig_month)

st.subheader('Predicted AQI by Day (Selected Month)')

# Create a selectbox for the day within the chosen month
day_input = st.selectbox(
    'Select Day',
    range(1, future_forecast_month['ds'].dt.day.max() + 1)
)

# Filter the forecast for the selected day
selected_day_forecast = future_forecast_month[future_forecast_month['ds'].dt.day == day_input]

if not selected_day_forecast.empty:
    st.write(selected_day_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Predicted AQI', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}))

    fig_day, ax_day = plt.subplots(figsize=(8, 4))
    ax_day.plot(selected_day_forecast['ds'], selected_day_forecast['yhat'], marker='o', linestyle='-', label='Predicted AQI')
    ax_day.fill_between(selected_day_forecast['ds'], selected_day_forecast['yhat_lower'], selected_day_forecast['yhat_upper'], color='k', alpha=.2, label='Prediction Interval')
    ax_day.set_xlabel('Date')
    ax_day.set_ylabel('AQI')
    ax_day.set_title(f'Predicted AQI for {pd.to_datetime(f"{year_input}-{month_input}-{day_input}").strftime("%B %d, %Y")}')
    ax_day.legend()
    st.pyplot(fig_day)
else:
    st.write("No data available for the selected day.")


# --- Weekly Analysis ---
st.subheader('Weekly Analysis')

# Group by week and calculate the mean predicted AQI
future_forecast_month['week'] = future_forecast_month['ds'].dt.isocalendar().week
weekly_avg_forecast = future_forecast_month.groupby('week')['yhat'].mean().reset_index()
weekly_avg_forecast['week'] = 'Week ' + weekly_avg_forecast['week'].astype(str)

st.write("Average Predicted AQI by Week for the Selected Month:")
st.write(weekly_avg_forecast)

fig_weekly, ax_weekly = plt.subplots(figsize=(10, 5))
ax_weekly.bar(weekly_avg_forecast['week'], weekly_avg_forecast['yhat'])
ax_weekly.set_xlabel('Week')
ax_weekly.set_ylabel('Average Predicted AQI')
ax_weekly.set_title(f'Average Predicted AQI by Week for {pd.to_datetime(f"{year_input}-{month_input}-01").strftime("%B %Y")}')
plt.xticks(rotation=45)
st.pyplot(fig_weekly)


# --- Multi-Line Graph (Overall AQI Trends with Selected Year Highlighted) ---
st.subheader('Overall AQI Trends with Selected Year Highlighted')
st.write("Multi-Line Graph: Displays AQI across all years in the dataset, highlighting the selected year.")

# Combine historical and forecasted data for the overall trend (using original scale)
# We need to generate predictions for the entire historical period to get the 'yhat' values
future_all_history = model.make_future_dataframe(periods=0, include_history=True)
forecast_all_history = model.predict(future_all_history)[['ds', 'yhat']]
forecast_all_history['yhat'] = np.expm1(forecast_all_history['yhat']) # Inverse transform historical predictions

# Merge with original historical data (not log-transformed 'y')
merged_df_all_history = pd.merge(df_original_y[['ds', 'y']].rename(columns={'y': 'Actual AQI'}),
                                 forecast_all_history[['ds', 'yhat']], on='ds')

merged_df_all_history['year'] = merged_df_all_history['ds'].dt.year

# Create a column to differentiate the selected year for highlighting
merged_df_all_history['highlight'] = merged_df_all_history['year'] == year_input

fig_multiline_overall = px.line(merged_df_all_history, x='ds', y='Actual AQI', color='highlight',
                                title='Overall AQI Trends with Selected Year Highlighted (Historical Data)')

# Add predicted line for the selected year if it's a historical year
if year_input in df_original_y['ds'].dt.year.unique():
     fig_multiline_overall.add_trace(go.Scatter(x=merged_df_all_history[merged_df_all_history['year'] == year_input]['ds'],
                                                y=merged_df_all_history[merged_df_all_history['year'] == year_input]['yhat'],
                                                mode='lines',
                                                name=f'Predicted AQI ({year_input})',
                                                line=dict(dash='dash')))

fig_multiline_overall.update_layout(xaxis_title='Date', yaxis_title='AQI')
st.plotly_chart(fig_multiline_overall)


# --- Donut Chart (AQI Category Distribution) ---
st.subheader('AQI Category Distribution')
st.write("A donut chart illustrating the distribution of different Air Quality Index (AQI) categories over the historical period.")

# Define AQI categories (example thresholds)
def get_aqi_category(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

# Apply category mapping to original historical data for the donut chart
df_original_y['AQI_Category'] = df_original_y['y'].apply(get_aqi_category) # Corrected function name

category_counts = df_original_y['AQI_Category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

fig_donut = px.pie(category_counts, values='Count', names='Category', hole=.3, title='Distribution of AQI Categories')
st.plotly_chart(fig_donut)


# --- Heatmap (Annual Air Quality Insights) ---
st.subheader('Annual Air Quality Insights')
st.write("Heatmap displaying annual air quality insights, focusing on monthly AQI for 2025 and the number of days in different AQI categories from 2020 to 2025.")

# Monthly AQI for 2025 (using predicted values)
future_2025_monthly = pd.DataFrame({'ds': pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')})
forecast_2025 = model.predict(future_2025_monthly)[['ds', 'yhat']]
forecast_2025['yhat'] = np.expm1(forecast_2025['yhat']) # Inverse transform 2025 monthly predictions
forecast_2025['month'] = forecast_2025['ds'].dt.month_name()
forecast_2025['year'] = forecast_2025['ds'].dt.year
monthly_avg_2025 = forecast_2025.groupby('month')['yhat'].mean().reset_index()
# Reorder months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_avg_2025['month'] = pd.Categorical(monthly_avg_2025['month'], categories=month_order, ordered=True)
monthly_avg_2025 = monthly_avg_2025.sort_values('month')


st.write("Average Predicted Monthly AQI for 2025:")
fig_heatmap_2025 = go.Figure(data=go.Heatmap(
        z=[monthly_avg_2025['yhat'].tolist()],
        x=monthly_avg_2025['month'].tolist(),
        y=['Average AQI'],
        colorscale='Hot',
        colorbar=dict(title='Average AQI')
    ))
fig_heatmap_2025.update_layout(title='Predicted Average Monthly AQI for 2025')
st.plotly_chart(fig_heatmap_2025)


# Number of days in each AQI category by year (2020-2025)
# Use original historical data (not log-transformed) for categories
historical_data_for_heatmap = df_original_y[df_original_y['ds'].dt.year >= 2020].copy()
historical_data_for_heatmap['year'] = historical_data_for_heatmap['ds'].dt.year
historical_data_for_heatmap['AQI_Category'] = historical_data_for_heatmap['y'].apply(get_aqi_category) # Corrected function name


future_dates_2024_2025 = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
future_df_2024_2025 = pd.DataFrame({'ds': future_dates_2024_2025})
forecast_2024_2025 = model.predict(future_df_2024_2025)[['ds', 'yhat']]
forecast_2024_2025['yhat'] = np.expm1(forecast_2024_2025['yhat']) # Inverse transform 2024-2025 predictions for category calculation
forecast_2024_2025['year'] = forecast_2024_2025['ds'].dt.year
forecast_2024_2025['AQI_Category'] = forecast_2024_2025['yhat'].apply(get_aqi_category) # Corrected function name

combined_heatmap_data = pd.concat([
    historical_data_for_heatmap[['year', 'AQI_Category']],
    forecast_2024_2025[['year', 'AQI_Category']]
])

category_day_counts = combined_heatmap_data.groupby(['year', 'AQI_Category']).size().unstack(fill_value=0)

# Ensure all categories are present even if count is 0 for a year
all_categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
for category in all_categories:
    if category not in category_day_counts.columns:
        category_day_counts[category] = 0

category_day_counts = category_day_counts[all_categories] # Ensure consistent order


st.write("Number of Days in Each AQI Category (2020-2025):")
fig_heatmap_categories = go.Figure(data=go.Heatmap(
        z=category_day_counts.values.tolist(),
        x=category_day_counts.columns.tolist(),
        y=category_day_counts.index.tolist(),
        colorscale='Viridis',
        colorbar=dict(title='Number of Days')
    ))
fig_heatmap_categories.update_layout(title='Number of Days in Each AQI Category (2020-2025)',
                                     xaxis_title='AQI Category',
                                     yaxis_title='Year')
st.plotly_chart(fig_heatmap_categories)


# --- Trend Analysis (Highest and Lowest) ---
st.subheader('Trend Analysis')

# Find overall highest and lowest historical AQI (using original scale)
highest_aqi_hist = df_original_y['y'].max()
date_highest_hist = df_original_y[df_original_y['y'] == highest_aqi_hist]['ds'].iloc[0].strftime('%Y-%m-%d')
lowest_aqi_hist = df_original_y['y'].min()
date_lowest_hist = df_original_y[df_original_y['y'] == lowest_aqi_hist]['ds'].iloc[0].strftime('%Y-%m-%d')


st.write(f"**Highest Historical AQI:** {highest_aqi_hist:.2f} on {date_highest_hist}")
st.write(f"**Lowest Historical AQI:** {lowest_aqi_hist:.2f} on {date_lowest_hist}")

# Find highest and lowest predicted AQI for the selected year (using inverse transformed values)
if year_input not in df_original_y['ds'].dt.year.unique(): # Only show for future predicted years
    highest_aqi_pred_year = forecast_year_pred['yhat'].max()
    date_highest_pred_year = forecast_year_pred[forecast_year_pred['yhat'] == highest_aqi_pred_year]['ds'].iloc[0].strftime('%Y-%m-%d')
    lowest_aqi_pred_year = forecast_year_pred['yhat'].min()
    date_lowest_pred_year = forecast_year_pred[forecast_year_pred['yhat'] == lowest_aqi_pred_year]['ds'].iloc[0].strftime('%Y-%m-%d')
    st.write(f"**Highest Predicted AQI ({year_input}):** {highest_aqi_pred_year:.2f} on {date_highest_pred_year}")
    st.write(f"**Lowest Predicted AQI ({year_input}):** {lowest_aqi_pred_year:.2f} on {date_lowest_pred_year}")

# Find highest and lowest predicted AQI for the selected month (using inverse transformed values)
highest_aqi_pred_month = future_forecast_month['yhat'].max()
date_highest_pred_month = future_forecast_month[future_forecast_month['yhat'] == highest_aqi_pred_month]['ds'].iloc[0].strftime('%Y-%m-%d')
lowest_aqi_pred_month = future_forecast_month['yhat'].min()
date_lowest_pred_month = future_forecast_month[future_forecast_month['yhat'] == lowest_aqi_pred_month]['ds'].iloc[0].strftime('%Y-%m-%d')
st.write(f"**Highest Predicted AQI ({pd.to_datetime(f'{year_input}-{month_input}-01').strftime('%B %Y')}):** {highest_aqi_pred_month:.2f} on {date_highest_pred_month}")
st.write(f"**Lowest Predicted AQI ({pd.to_datetime(f'{year_input}-{month_input}-01').strftime('%B %Y')}):** {lowest_aqi_pred_month:.2f} on {date_lowest_pred_month}")

