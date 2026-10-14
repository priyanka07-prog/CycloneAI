# 🌪️ CycloneAI — Tropical Cyclone Intensity Classification

An AI/ML-based project for classifying tropical cyclone intensity using historical cyclone observations. CycloneAI combines machine learning, historical data visualization, and an interactive dashboard to explore cyclone patterns and classify observed storm intensity.

## 📌 Project Overview

Tropical cyclones can cause significant damage to coastal regions and communities. Understanding their intensity is important for studying historical storm behavior.

**CycloneAI** uses historical cyclone data and a Random Forest classifier to categorize cyclone observations into three intensity classes based on wind speed.

The project includes a Python-based dashboard for exploring historical cyclone observations and testing intensity predictions using selected input features.

> **Note:** This project is for educational and research purposes. It classifies observed intensity and does not forecast future cyclone intensity, movement, or landfall.

## 🎯 Objectives

- Analyze historical tropical cyclone observations.
- Preprocess cyclone data for machine learning.
- Classify cyclone intensity into three categories.
- Evaluate model performance using standard classification metrics.
- Visualize historical cyclone observations.
- Provide an interactive dashboard for model predictions.

## 📊 Dataset

**Dataset:** IBTrACS — International Best Track Archive for Climate Stewardship.

IBTrACS provides historical tropical cyclone best-track observations collected from meteorological agencies around the world.

Dataset source: https://www.ncei.noaa.gov/products/international-best-track-archive

The project uses historical observations, including:

- Latitude (`LAT`)
- Longitude (`LON`)
- Month (`MONTH`)
- Central pressure (`WMO_PRES`, where available)
- Wind speed (`WMO_WIND`, where available)

The dataset is stored locally in `data/raw/ibtracs.csv`.

## 🛠️ Technologies Used

- **Python** — primary programming language
- **Pandas and NumPy** — data processing
- **Scikit-learn** — machine learning and model evaluation
- **Joblib** — saving and loading the trained model
- **Plotly** — interactive visualizations
- **Dash** — interactive web dashboard
- **Dash Bootstrap Components** — dashboard layout and styling

## 🤖 Machine Learning Model

CycloneAI uses a **Random Forest Classifier** with a median-value imputer to handle missing feature values.

### Input features

- Latitude
- Longitude
- Month
- Central atmospheric pressure

### Target classes

| Class | Wind speed |
|---|---|
| Lower intensity | Below 34 knots |
| Tropical-storm-force | 34 to below 64 knots |
| Hurricane-force | 64 knots or higher |

These are project-defined intensity categories based on wind-speed thresholds, not a complete official cyclone classification system for every ocean basin.

### Model configuration

- Algorithm: Random Forest Classifier
- Number of trees: 200
- Class weighting: Balanced
- Random state: 42
- Missing-value handling: Median imputation
- Evaluation strategy: 80/20 group-based train-test split, grouped by cyclone ID

Grouping observations by cyclone ID helps prevent observations from the same cyclone from appearing in both the training and test sets.

## 📈 Model Performance

The trained model achieved the following results on the held-out test set.

| Metric | Result |
|---|---:|
| Test accuracy | 86.4% |
| Macro F1-score | 0.87 |
| Test observations | 24,639 |
| Training observations | 96,337 |

### Classification Report

| Intensity class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Hurricane-force | 0.92 | 0.89 | 0.90 |
| Lower intensity | 0.84 | 0.84 | 0.84 |
| Tropical-storm-force | 0.85 | 0.86 | 0.86 |

### Confusion Matrix

Rows represent actual classes, and columns represent predicted classes.

Class order: Hurricane-force, Lower intensity, Tropical-storm-force.

| Actual / Predicted | Hurricane-force | Lower intensity | Tropical-storm-force |
|---|---:|---:|---:|
| Hurricane-force | 5,399 | 100 | 572 |
| Lower intensity | 68 | 6,419 | 1,116 |
| Tropical-storm-force | 416 | 1,083 | 9,466 |

The diagonal values represent correctly classified observations. The remaining values represent misclassifications.

**Interpretation:** The model performs reasonably well as a baseline classifier, but some observations are assigned to the wrong intensity category. In particular, missing stronger storms is an important risk to consider.

## 📊 Dashboard Features

CycloneAI provides an interactive dashboard featuring:

- Historical cyclone observation visualization
- Wind-intensity color scale
- Model performance summary cards
- Latitude and longitude inputs
- Month and atmospheric pressure inputs
- Intensity prediction results

The prediction form uses the trained model to classify an observation based on the selected input features.

## 📁 Project Structure

```text
CycloneAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   └── ibtracs.csv
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_analysis.py
│   ├── visualization.py
│   └── train_model.py
├── artifacts/
│   └── cyclone_intensity_model.joblib
└── assets/
    └── custom.css
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11
- Anaconda or Miniconda
- Visual Studio Code (recommended)

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CycloneAI
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Activate the environment

If your `cyclone-ai` environment already exists:

```bash
conda activate cyclone-ai
```

Otherwise, create it:

```bash
conda create -n cyclone-ai python=3.11
conda activate cyclone-ai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset and model

Ensure the IBTrACS CSV file is available at `data/raw/ibtracs.csv` and the trained model exists at `artifacts/cyclone_intensity_model.joblib`.

If the trained model is not available, train it using the project's training script after placing the dataset correctly:

```bash
python src/train_model.py
```

### 5. Run the dashboard

```bash
python app.py
```

Open the local URL printed in the terminal. For a typical Dash application, this is `http://127.0.0.1:8050/`.

## 🧪 Testing and Evaluation

The project was evaluated using a held-out test set and the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The prediction interface was tested using sample latitude, longitude, month, and pressure inputs.

Further testing with independent storms, different time periods, and additional environmental variables would help assess generalization.

## ⚠️ Limitations

- The model classifies observed cyclone intensity; it does not predict future intensity or storm trajectories.
- Predictions depend on a limited set of input features.
- Historical observations may contain missing values or measurement inconsistencies.
- An accuracy score alone does not establish operational reliability.
- The project is not designed for emergency decisions or official weather warnings.

## 🚀 Future Improvements

- Incorporate sea-surface temperature and wind-shear data.
- Compare Random Forest with other machine learning models.
- Evaluate performance across independent years and cyclone basins.
- Improve historical storm-track visualization.
- Develop a separate forecasting model using time-series observations.
- Deploy the dashboard for public demonstration.

## 🎓 Project Purpose

CycloneAI is a learning and portfolio project demonstrating data preprocessing, supervised machine learning, model evaluation, and interactive data visualization using Python.


---

*Built with Python and machine learning for educational and research purposes.*