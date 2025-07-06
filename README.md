# What Makes a Hit? Predicting Spotify Streams with Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the analysis for the "Scoring and Machine Learning Project" completed for Université Paris 1 Panthéon-Sorbonne. It explores the factors that drive the popularity of songs on Spotify and uses machine learning to predict stream counts.

## Overview

The primary goal of this project is to identify the key drivers behind a song's streaming success on Spotify. Using a dataset of the most popular songs from 2023, this analysis leverages several machine learning models to answer the question: "What makes a hit?"

The project covers a full data science workflow:
*   **Data Cleaning and Preparation**: Handling missing values, correcting data types, and feature scaling.
*   **Exploratory Data Analysis (EDA)**: Investigating relationships between song attributes (like `danceability`, `valence_%`) and platform metrics (like playlist and chart inclusions) with stream counts.
*   **Feature Engineering**: Using Variance Inflation Factor (VIF) to detect and mitigate multicollinearity.
*   **Model Implementation**: Training and evaluating four regression models:
    1.  LASSO Regression
    2.  Decision Tree
    3.  Random Forest
    4.  Gradient Boosting
*   **Model Evaluation**: Comparing models using R-squared, MAE, MSE, and RMSE to identify a champion model.

## Key Findings

The Gradient Boosting model emerged as the top performer, explaining **79.19%** of the variance in stream counts.

*   **Playlist inclusion is the most critical factor**: A song's presence on platforms like Deezer and Apple Music is the strongest predictor of high stream counts.
*   **Emotional tone matters**: The `valence_%` (a measure of musical positiveness) was a significant secondary driver of streams.
*   **Chart performance provides a secondary boost**: While less impactful than playlists, inclusions in Spotify and Shazam charts also contribute to a song's success.

| Metric | LASSO | Decision Tree | Random Forest | **Gradient Boosting** |
| :--- | :---: | :---: | :---: | :---: |
| **R² (%)** | 72.08 | 68.71 | 77.86 | **79.19** |
| **MAE** | 211.31 | 205.31 | 181.21 | **179.52** |
| **RMSE** | 303.85 | 322.44 | 271.19 | **262.92** |

## How to Run This Project

To replicate this analysis, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/spotify-hit-predictor.git
    cd spotify-hit-predictor
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the analysis script:**
    Execute the main Python script from the root of the project directory. The script will perform the full data analysis, train the models, and save all generated figures to the `reports/figures/` folder.
    ```bash
    python src/main.py
    ```

## Authors
*   Jessie Cameron
*   Gabriela Moravcikova
*   Kateryna Zaichenko