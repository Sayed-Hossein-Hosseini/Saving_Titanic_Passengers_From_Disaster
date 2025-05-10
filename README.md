# 🚢 Saving_Titanic_Passengers_From_Disaster! 🚀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sayed-Hossein-Hosseini/Saving_Titanic_Passengers_From_Disaster/blob/master/Saving_Titanic_Passengers_From_Disaster.ipynb)

Welcome aboard this Titanic data adventure! This repository hosts a Jupyter Notebook where we dive deep into the Titanic passenger dataset. The mission? To build a Logistic Regression model *from scratch* and predict who made it out alive! 🎲

## 🗺️ Project Overview

The core goal is to explore the iconic Titanic dataset, wrangle the data through preprocessing and feature engineering, and ultimately, train our very own custom Logistic Regression model. This model will then classify passengers into two fateful categories: "Survived" 🥳 or "Not Survived" 😔.

## 🧭 Notebook Journey & Milestones

The `Saving_Titanic_Passengers_From_Disaster.ipynb` notebook will guide you through the following exciting stages:

1.  **📚 Loading Libraries:**
    *   Importing the trusty Python crew: `pandas` for data voyages, `numpy` for numerical wizardry, `matplotlib` & `seaborn` for stunning visuals, and `scikit-learn` for some handy preprocessing tools.

2.  **📥 Data Embarkation:**
    *   Hoisting the training (`train.csv`) and test (`test.csv`) datasets aboard from the `/content/Titanic/` dock.
    *   A quick peek at the first few passengers (rows) of both datasets.

3.  **📊 Data Discovery & EDA (Part 1):**
    *   Charting the data's dimensions (rows and columns).
    *   Gathering intel on columns: non-null counts and data types (`.info()`).
    *   Unveiling descriptive stats for numerical features (`.describe()`).
    *   Peeking at descriptive stats for categorical (object) features.
    *   Investigating and reporting missing treasure (NaN values) in each column.

4.  **🎨 Data Visualization (EDA - Part 2):**
    *   **Numerical Features:**
        *   Plotting histograms and KDEs to map data distributions.
        *   Charting Box Plots to spot any rogue waves (outliers).
        *   Creating Pair Plots to explore relationships between numerical features (if feature count allows).
    *   **Categorical Features:**
        *   Crafting Count Plots to display category frequencies (skipping high-cardinality features like `Name`, `Ticket`, `Cabin`).

5.  **🔗 Correlation & Initial Feature Scouting:**
    *   Calculating and visualizing the correlation matrix for numerical features (training data).
    *   Plotting a heatmap to reveal these correlations.
    *   Identifying and flagging numerical features with low correlation to our target (`Survived`) for potential removal (threshold 0.01). `PassengerId` is spotted here!
    *   Manually jettisoning less useful or redundant columns (`Ticket`, `Cabin`, `PassengerId`, `Name`) from both datasets.

6.  **🩹 Patching Missing Data:**
    *   **Numerical Features:** Filling NaN-holes in `Age` and `Fare` with the *mean* value from the training set.
    *   **Categorical Features:** Plugging gaps in `Embarked` with the *mode* (most common value) from the training set.
    *   Final inspection: No NaNs left behind!

7.  **🛠️ Preparing Data for the Model:**
    *   Separating features (`X_train_orig`) from the target (`y_train_orig`) in the training data.
    *   Identifying the remaining numerical and categorical sailors (columns).
    *   **Standardization:** Applying `StandardScaler` to numerical features.
    *   **One-Hot Encoding:** Deploying `OneHotEncoder` (with `handle_unknown='ignore'`) for categorical features.
    *   Using `ColumnTransformer` to apply these transformations smartly.
    *   Converting our polished data into NumPy arrays, ready for action!

8.  **🤖 Building & Training Logistic Regression (From Scratch!):**
    *   Crafting the `LogisticRegressionScratch` class, featuring:
        *   The mighty Sigmoid function (`_sigmoid`).
        *   The insightful Cost function (Log Loss) (`_cost_function`) with a dash of epsilon for stability.
        *   The learning Fit function (`fit`) powered by Gradient Descent.
        *   Probability (`predict_proba`) and class (`predict`) prediction functions.
        *   An optional intercept term for extra flexibility.
    *   Training our custom model on the preprocessed training data.
    *   Plotting the cost history to witness the model's learning journey (convergence).

9.  **🔮 Prediction & Crafting the Output:**
    *   Unleashing our trained model on the preprocessed test data.
    *   Assembling a Kaggle-ready DataFrame (with `PassengerId` and `Survived` predictions).
    *   Saving the precious predictions to `Saving_Titanic_Passengers_From_Disaster.csv`.

## 🚀 How to Launch This Project

1.  Clone this repository or download the notebook (`.ipynb`) and data files.
2.  Ensure `train.csv` and `test.csv` are in a folder named `Titanic` located at `/content/` (if using Colab) or an equivalent path locally. Adjust file paths in the "Data Loading" cell if needed.
3.  Open the notebook in your favorite Jupyter environment (Jupyter Lab, Jupyter Notebook, or Google Colab).
4.  Run the cells in order and watch the magic unfold!

## 📊 Data Source

The dataset for this voyage comes from the legendary "Titanic - Machine Learning from Disaster" competition on Kaggle.

## 🛠️ Libraries on Deck

*   `pandas`
*   `numpy`
*   `seaborn`
*   `matplotlib`
*   `scikit-learn`
*   `io`
*   `warnings`

## ✨ Key Ingredients of Our Custom Logistic Regression

*   **`__init__(...)`:** The captain's constructor.
*   **`_add_intercept(...)`:** Adds the essential intercept term.
*   **`_sigmoid(...)`:** The S-shaped heart of logistic regression.
*   **`_cost_function(...)`:** Measures how well (or not so well) we're doing.
*   **`fit(...)`:** The training drill sergeant (Gradient Descent).
*   **`predict_proba(...)`:** Peeks into the future (probabilities).
*   **`predict(...)`:** Makes the final call (class predictions).

## 📄 Output File

Upon successful execution, you'll find `Saving_Titanic_Passengers_From_Disaster.csv` in your directory, containing:
*   `PassengerId`: Passenger identifiers.
*   `Survived`: Our model's survival predictions (0 = No, 1 = Yes).

Perfect for your Kaggle submission! 🏆

## 💡 Potential Future Expeditions (Improvements)

*   **Cross-validation:** For a more seaworthy model evaluation.
*   **Hyperparameter Tuning:** Fine-tuning our logistic regression's compass (learning rate, iterations).
*   **Advanced Feature Engineering:** Extracting titles, crafting family size features, and more!
*   **Exploring Other Models:** Charting new waters with Decision Trees, Random Forests, Gradient Boosting, or even Neural Networks.
*   **Error Analysis:** Investigating where our predictions went adrift.

## ✒️ Author

*   **Name:** \[Sayyed Hossein Hosseini DolatAbadi]
*   **Email:** \[S.Hossein.Hosseini1381@gmail.com]

## 📜 License

This project is licensed under the MIT License.
