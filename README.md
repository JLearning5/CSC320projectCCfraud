# Credit-Card-Fraud-Detection-Project

This project aims to detect fraudulent credit card transactions using a Logistic Regression model. The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle.

## Voilà

You can run this project interactively in your browser via Voilà by clicking the badge below:

[![Launch Voilà](https://img.shields.io/badge/launch-Voilà-blue.svg)](https://mybinder.org/v2/gh/JLearning5/Credit-Card-Fraud-Detection-project/HEAD?urlpath=voila/render/notebooks/Fraud_Detection_preview.ipynb)

## Binder

You can run the notebooks in your browser via Binder by clicking the badge below:

[![Binder](https://notebooks.gesis.org/binder/jupyter/user/jlearning5-cred-tection-project-7x6damc7/lab/workspaces/auto-G/tree/notebooks/Fraud_detection.ipynb)

## Project Overview

The goal of this project is to build a machine learning model to detect fraudulent transactions. The dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent.

## Features

- **Data Preprocessing:** Handling imbalanced data using under-sampling.
- **Model Training:** Logistic Regression model for classification.
- **Evaluation:** Model evaluation using accuracy score, ROC curve, and precision-recall curve.
- **Visualization:** Various plots to visualize class distribution, transaction amounts, time distribution, correlation matrix, and more.

## Dependencies

The project uses the following libraries:

- Python 3.9
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Ipywidgets
- JupyterLab
- Voila

These dependencies are listed in the `environment.yml` file for easy setup using conda.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/heinerigel/Credit-Card-Fraud-Detection-project.git
   cd Credit-Card-Fraud-Detection-project
