# Project3_Machine_Learning
Machine learning project of an e-commerce economic and geographic dataset

# Description
A machine learning project was developed using a dataset of economic and geographic values from an e-commerce platform, extracted from archive.ics.uci.edu. 

A Streamlit App was created that includes an analysis with different models in the file `ecommerceML.py`:

#### Model Classification

**Classification and Regression Models**:
- **K-Nearest Neighbors (KNN)**: Used for both classification and regression.
- **Linear Regression**: Used for regression.
- **Logistic Regression**: Used for classification.
- **Decision Tree**: Used for both classification and regression.

**Ensemble Models**:
- **Bagging**: Used to create multiple versions of a predictor and combine their results. Example: Bagged Decision Trees.
- **Random Forest**: An improvement of Bagging applied to decision trees.
- **AdaBoost**: A boosting method that adjusts the weights of observations based on previous errors.
- **Gradient Boosting**: A boosting technique that optimizes a generic loss function.

#### Optimization with Grid Search and Random Search

- **Grid Search**
- **Random Search**

In the file `ecommerceML_hyper_bal_graf.py`, in addition to the above, balancing techniques are included if the App detects that they are necessary:

#### Data Balancing Techniques

Balancing data is crucial when working with imbalanced datasets (where one class is significantly more numerous than another). Common techniques include:

- **Under-sampling**: Reducing the number of examples of the majority class.
- **Over-sampling**: Increasing the number of examples of the minority class.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generating synthetic examples of the minority class.

Parameters can be selected by the user in advance, deciding on the target and feature columns that can be trained with the models.

# Sources from which it has been extracted

[https://github.com/data-bootcamp-v4/lessons/tree/main/5_6_eda_inf_stats_tableau/project/files_for_project](https://archive.ics.uci.edu/dataset/352/online+retail)

# Folder structure

The person interested in the analysis will find the following documents in the project folder:

- ecommerceML.py
- ecommerceML_hyper_bal_graf.py

The person interested in analyzing the development of the project corpus should run the various code cells in the order in which they appear in .py file.


# Sample screenshots
![Captura de pantalla 2024-05-24 a las 18 21 18](https://github.com/Kabuto4dev/Project3_Machine_Learning/assets/100389319/85c8c663-b3a9-49a9-bfe7-cf734699644c)
![Captura de pantalla 2024-05-24 a las 18 21 05](https://github.com/Kabuto4dev/Project3_Machine_Learning/assets/100389319/4836b4ed-a903-418f-afda-d5dbc050d8f8)
![Captura de pantalla 2024-05-24 a las 18 18 25](https://github.com/Kabuto4dev/Project3_Machine_Learning/assets/100389319/b61eeee6-8d6e-4a88-b300-6be2e847a933)

# Copyright

Proprietary software license. Contact us at +34 641 024 603, to request the use or testing of the code.

# Authors

Juan Fran Sánchez contacto@juanfransf.com

# Contribution: 

If you would like to contribute to the development of the project please contact: 
contacto@juanfransf.com - +34 641 024 603
