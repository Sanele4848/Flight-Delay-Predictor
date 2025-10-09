# Predicting Flight Delays Using Machine Learning

-----

## Project Overview

This project presents an end-to-end machine learning system for predicting flight delay rates for U.S. domestic flights. The system analyzes historical data to identify patterns related to airlines, airports, and seasonality, providing actionable insights for travelers and industry stakeholders.

The final model, a **Random Forest Regressor**, achieved a **Mean Absolute Error of 2.12%** and an **R² score of 88.88%** on the test set, demonstrating high accuracy in predicting delay probabilities.

The project includes a full data processing and model training pipeline, as well as a deployed interactive web application for real-time predictions.

-----

## Project Structure

The project is organized into a modular structure for clarity and maintainability:

```
.
├── models/
│   └── # Directory for saved model artifacts
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── prediction_pipeline.py
├── app/
│   └── Flight_Delay_app.py
├── main.py
├── requirements.txt
└── README.md
```

-----

## Dataset

The project uses the "**Flight Delay Data for U.S. Airports by Carrier August 2013 - August 2023**" dataset, which is sourced from the Bureau of Transportation Statistics (BTS).

  * **Source**: Kaggle
  * **Description**: The dataset contains monthly aggregated data on flight arrivals, delays, cancellations, and delay causes for major U.S. airlines across numerous airports.

> **Note**: To run the pipeline, please download the **Airline\_Delay\_Cause.csv** file from the source and place it in a `data/` directory at the root of the project.

-----

## Setup and Installation

Follow these steps to set up the project environment. A Python version of 3.8 or higher is recommended.

### Step 1: Clone the Repository

Download and unzip the project files to your local machine.

### Step 2: Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies. Open a terminal in the project's root directory and run:

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

With your virtual environment activated, install all the required libraries using the **requirements.txt** file:

```bash
pip install -r requirements.txt
```

-----

## Usage

There are two main ways to interact with this project: running the full training pipeline or launching the interactive web application.

### To Run the Full Training Pipeline

This will execute all steps: data loading, cleaning, feature engineering, model training, evaluation, and saving the final model artifacts to the `models/` directory.

From the root directory, run the following command in your terminal:

```bash
python main.py
```

### To Launch the Streamlit Web Application

This will start a local web server and open the interactive prediction tool in your browser. Ensure that the model artifacts already exist in the `models/` folder (by running **main.py** at least once).

From the root directory, run the following command in your terminal:

```bash
streamlit run app/Flight_Delay_app.py
```

You can then navigate to the local URL provided in the terminal to use the application.

-----

## Core Dependencies

The main libraries used in this project are:

  * **pandas & numpy**: For data manipulation and numerical operations.
  * **scikit-learn**: For machine learning modeling and preprocessing.
  * **streamlit**: For building the interactive web application.
  * **plotly**: For creating interactive data visualizations.
  * **joblib**: For saving and loading the trained model.

A complete list is available in the **requirements.txt** file.