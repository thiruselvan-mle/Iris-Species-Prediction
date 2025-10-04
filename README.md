# 🌸 Iris Species Prediction

A Machine Learning project to classify **Iris flowers** into three species — **Setosa, Versicolor, Virginica** — based on their petal and sepal measurements.  
Built with **Scikit-Learn, Streamlit, Pandas, Matplotlib, Seaborn**.

---

## 📂 Project Structure
```bash
 Iris-Species-Prediction/
 ├── data/ # dataset (raw & processed)
 ├── notebooks/ # Jupyter notebooks (exploration → evaluation)
 ├── src/ # helper Python scripts
 ├── models/ # saved models & encoders
 ├── app/ # Streamlit app
 ├── README.md
 └── requirements.txt
 bash

---

## 📊 Workflow

1. **Data Exploration** – Checked dataset structure (`df.info()`, `head()`, `describe()`).
2. **Data Cleaning** – Handled missing values, ensured correct datatypes.
3. **Exploratory Data Analysis (EDA)** – Visualized distributions and correlations using **Seaborn & Matplotlib**.
4. **Model Training** – Trained multiple algorithms (SVC, RandomForest, Logistic Regression) and compared results.
5. **Model Evaluation** – Selected **SVC** as best model (100% accuracy on test set, AUC = 1.0).
6. **Deployment** – Built an interactive **Streamlit web app** for real-time predictions.

---
