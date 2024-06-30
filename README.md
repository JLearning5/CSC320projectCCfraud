# Credit Card Fraud Detection Project

This project aims to detect fraudulent credit card transactions using a Logistic Regression model. The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle.

Click the badge below to view the dataset:

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

## How to Access and Use the Fraud Detection Web Page

For detailed instructions on how to access and use the web page, please refer to the User Guide:

[![User Guide](https://img.shields.io/badge/User_Guide-PDF-blue)](https://drive.google.com/file/d/1UAq9wCanIDme9FpdijMJFmLMpHiIYrb-/view?usp=sharing)

## Google Colaboratory

You can run this project interactively in your browser via Google Colab by clicking the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DsxUjbLdVjkHThyTxdl35Kbo7Ki33NlV?usp=sharing)

## How to Access the Machine Learning Workflow

To access and examine the machine learning workflow, please click the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pTziVNjo7v9jFVEqm2ynnSvbHcVrZ3fb?usp=sharing)

## Project Overview

The goal of this project is to build a machine learning model to detect fraudulent transactions. The dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent.

## Features

- **Data Preprocessing:** Handling imbalanced data using over-sampling.
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
- Imbalanced-learn

These dependencies are listed in the `requirements.txt` file.

### Installation

1. Clone this repository:
   ```bash
   git clone https://https://github.com/JLearning5/Credit-Card-Fraud-Detection-project.git
   cd Credit-Card-Fraud-Detection-project
   
2. Create a virtual environment to manage dependencies:
   ```bash
   python -m venv env

3. Activate the virtual environment:

   On Windows:
   ```bash
   .\env\Scripts\activate
   ```
   
   On macOS and Linux:
   ```bash
   source env/bin/activate

4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   
### Running the Project

5. Start Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   ```

   or

   ```bash
   jupyter lab
   
6. Open the project notebook:
   
   Navigate to 'notebooks' directory in the Jupyter interface and open 'Fraud_detection.ipynb'.

### Running the Interactive Voilà Interface

7. To run the Voilà interface, ensure you have Voilà installed:
   ```bash
   pip install voila

8. Launch the Voilà interface:
   ```bash
   voila notebooks/Fraud_Detection_preview.ipynb
   ```
