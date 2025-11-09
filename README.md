# 🎯 CODSOFT Internship Projects - Machine Learning

## 👨‍💻 Author: **RISHIRAJ**
📅 **Submission: November 2025**  
🏫 **NIMS University, Jaipur**  
📘 **Domain:** Machine Learning Projects (Internship Tasks)

---

## 📂 Repository Overview
This repository contains three machine learning projects completed as part of the **CODSOFT Internship Program**.  
Each project explores key ML techniques such as **data preprocessing**, **feature engineering**, **model training**, **evaluation**, and **prediction** using real-world datasets from **Kaggle** via **KaggleHub**.

---

## 🚀 Tasks Overview

### 🧩 **Task 1: Titanic Survival Prediction**
**Dataset:** [Titanic Dataset (Kaggle)](https://www.kaggle.com/yasserh/titanic-dataset)

#### 🧠 Objective:
Predict whether a passenger survived the Titanic disaster based on features such as **age, gender, class, and fare**.

#### ⚙️ Process:
- Imported dataset using `kagglehub`
- Cleaned and handled missing values
- Encoded categorical variables (`Sex`, `Embarked`)
- Trained a **Logistic Regression** model
- Evaluated using **Accuracy**, **Confusion Matrix**, and **Classification Report**

#### 📈 Result:
Model achieved an accuracy of **~80–85%**, showing good prediction ability on unseen passenger data.

#### 🧮 Libraries Used:
`pandas`, `numpy`, `scikit-learn`, `kagglehub`

📁 *File:* `titanic_codsoft_task1.py`

---

### 🎬 **Task 2: Movie Rating Prediction**
**Dataset:** [IMDB India Movies (Kaggle)](https://www.kaggle.com/adrianmcmahon/imdb-india-movies)

#### 🧠 Objective:
Predict the **IMDB rating** of a movie based on its **genre, director, actors, runtime, and release year**.

#### ⚙️ Process:
- Loaded dataset via `kagglehub`
- Cleaned and selected important features
- Encoded text data (genre, director, actors)
- Trained a **Random Forest Regressor**
- Evaluated using **MAE**, **RMSE**, and **R² Score**

#### 📈 Result:
The model performs effectively with low MAE and high R², accurately estimating movie ratings.

#### 🧮 Libraries Used:
`pandas`, `numpy`, `scikit-learn`, `kagglehub`

📁 *File:* `movie_rating_prediction_codsoft_task2.py`

---

### 🌸 **Task 3: Iris Flower Classification**
**Dataset:** [Iris Flower Dataset (Kaggle)](https://www.kaggle.com/arshid/iris-flower-dataset)

#### 🧠 Objective:
Classify Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on sepal and petal measurements.

#### ⚙️ Process:
- Loaded dataset using `kagglehub`
- Dropped unnecessary columns
- Encoded target labels
- Trained a **Random Forest Classifier**
- Evaluated model performance on test data

#### 📈 Result:
Achieved an accuracy of **97–100%** on the test set, showing excellent classification performance.

#### 🧮 Libraries Used:
`pandas`, `numpy`, `scikit-learn`, `kagglehub`

📁 *File:* `iris_flower_classification_codsoft_task3.py`

---

## 🧰 Common Project Requirements
Each project uses the following core Python libraries:

