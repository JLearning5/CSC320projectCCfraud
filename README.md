# Credit Card Fraud Detection Project

This project aims to detect fraudulent credit card transactions using a Logistic Regression model. The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle.

## How to Access and Use the Fraud Detection Web Page

For detailed instructions on how to access and use the web page, please refer to the User Guide:

[![User Guide](https://img.shields.io/badge/User_Guide-PDF-blue)](https://github.com/yourusername/fraud-detection/raw/main/user_guide.pdf)

# Google Colaboratory

You can run this project interactively in your browser via Google Colab by clicking the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AChLOW-6Onf3h_xSq_urlGBCHGWjilCz?usp=sharing)

## Project Overview

The goal of this project is to build a machine learning model to detect fraudulent transactions. The dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent.

## Features

- **Data Preprocessing:** Handling imbalanced data using under-sampling.
- **Model Training:** Logistic Regression model for classification.
- **Evaluation:** Model evaluation using accuracy score, ROC curve, and precision-recall curve.
- **Visualisation:** Various plots to visualise class distribution, transaction amounts, time distribution, correlation matrix, and more.

## Dependencies

The project uses the following libraries:

- Python 3.9
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Ipywidgets

These dependencies are listed in the `requirements.txt` file for easy setup using conda.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/heinerigel/Credit-Card-Fraud-Detection-project.git
   cd Credit-Card-Fraud-Detection-project
