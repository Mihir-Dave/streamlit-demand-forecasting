# Project Report: Demand Forecasting Streamlit App

## 1. Introduction

Demand forecasting is an important technique used to estimate future product demand based on historical sales data. This project provides a web-based dashboard that helps analyze weekly demand and compare forecasting models.

## 2. Objective

The main objective of this project is to build a simple and interactive demand forecasting system that:

- Accepts a CSV demand dataset
- Visualizes historical sales trends
- Applies multiple forecasting algorithms
- Compares model performance using RMSE
- Displays the best-performing model

## 3. Technology Stack

- Python for backend logic and data processing
- Streamlit for the web application interface
- Pandas for dataset handling
- Matplotlib for plotting charts
- Statsmodels for Holt-Winters and ARIMA models
- Prophet for time-series forecasting
- Scikit-learn for RMSE evaluation

## 4. Methodology

The user uploads a CSV dataset containing weekly demand values. The application converts the date column into a time-series index and resamples the data weekly. The dataset is split into training and testing portions. Three forecasting models are trained and evaluated:

- Holt-Winters Exponential Smoothing
- ARIMA
- Prophet

The predictions are compared against the test data using Root Mean Squared Error. The model with the lowest RMSE is selected as the best forecasting model.

## 5. Functional Modules

### Dataset Upload

The application allows users to upload a `demand.csv` file.

### Data Preview

The first few rows of the uploaded dataset are displayed for verification.

### KPI Analysis

The dashboard calculates total sales, average weekly sales, and peak weekly sales.

### Forecasting Models

The application trains Holt-Winters, ARIMA, and Prophet models on the uploaded demand data.

### Model Comparison

RMSE values are calculated and displayed in a comparison table.

### Forecast Visualization

The app plots the original demand data and model predictions on a graph.

## 6. Results

The output includes:

- Weekly sales trend
- Model performance comparison
- Best forecasting model
- Forecast comparison chart

Add screenshots and output images in the `screenshot-output/` folder.

## 7. Conclusion

This project demonstrates how time-series forecasting models can be used to predict future demand. The Streamlit interface makes the system easy to use and helps users compare model performance visually.

## 8. Future Scope

- Add downloadable forecast reports
- Add support for monthly and daily forecasting
- Add interactive charts
- Improve model tuning options
- Deploy the project online

## 9. Team Members

- Mihir Dave
- Mehul Jain
