#  Student Stress & Burnout Predictor

A machine learning project that predicts student burnout risk (Low / Medium / High) based on lifestyle and academic survey data. Built as a BYOP (Bring Your Own Project) capstone for a Data Science / ML course.

---
##  Problem Statement

Student burnout is a growing concern in academic environments, yet it often goes undetected until it significantly impacts performance and well-being. This project collects real survey data from students and trains a classification model to predict an individual's burnout risk level — enabling early, data-driven intervention.

---

##  What This Project Does

- Collects student lifestyle and academic data via a Google Form survey
- Performs exploratory data analysis (EDA) to uncover stress patterns
- Trains and compares multiple ML classifiers (Logistic Regression, Random Forest, etc.)
- Evaluates model performance using accuracy, confusion matrix, and feature importance
- Outputs a prediction of burnout risk: **Low**, **Medium**, or **High**

---

##  Project Structure

```
student-burnout-predictor/
│
├── data/
│   └── survey_responses.csv        # Raw collected survey data
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Data cleaning & feature engineering
│   └── 03_model_training.ipynb     # Model training, evaluation & comparison
│
├── models/
│   └── burnout_classifier.pkl      # Saved trained model (joblib)
│
├── src/
│   └── predict.py                  # Script to run predictions on new input
│
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

##  Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/student-burnout-predictor.git
cd student-burnout-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in order: `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_model_training.ipynb`

---

##  Dataset

Data was collected via a **Google Form survey** shared with college students. Each response captures:

| Feature | Description |
|---|---|
| `sleep_hours` | Average hours of sleep per night |
| `study_hours` | Average daily study hours |
| `social_activity` | Frequency of social interactions (1–5 scale) |
| `physical_activity` | Exercise frequency per week |
| `academic_pressure` | Self-rated academic pressure (1–10) |
| `extracurricular_load` | Number of clubs/activities |
| `assignment_backlog` | Number of pending assignments |
| `mood_rating` | Self-rated mood over the past week (1–10) |
| `burnout_level` | **Target** — Low / Medium / High (self-reported) |

> **Note:** All responses are anonymous. No personally identifiable information was collected.

---

##  Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Baseline classifier, interpretable coefficients |
| Random Forest | Best performer; used for feature importance |
| K-Nearest Neighbors | Included for comparison |

Models are evaluated using **5-fold cross-validation**, accuracy score, and a confusion matrix.

---

##  Running a Prediction

After training the model (run notebook `03_model_training.ipynb` first), you can predict burnout risk for a new student profile:

```bash
python src/predict.py
```

You will be prompted to enter values for each feature. The model will output a predicted burnout risk level:

```
Enter sleep hours per night: 5
Enter daily study hours: 8
Enter social activity score (1-5): 2
...

Predicted Burnout Risk: HIGH
```

---

##  Results Summary

| Model | Cross-Val Accuracy |
|---|---|
| Logistic Regression | ~78% |
| Random Forest | ~85% |
| K-Nearest Neighbors | ~74% |

Top features influencing burnout (from Random Forest):
1. Academic pressure
2. Sleep hours
3. Assignment backlog
4. Mood rating

---

##  Key Findings

- Students sleeping fewer than 6 hours showed a significantly higher rate of **High** burnout classification.
- Academic pressure was the single strongest predictor of burnout level.
- Social activity acted as a protective factor — students with higher social engagement trended toward lower burnout risk.

---

##  Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
joblib
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

##  Limitations & Future Work

- Survey sample size is limited (~50–100 students from a single institution); results may not generalise broadly.
- Self-reported burnout labels introduce subjectivity bias.
- Future improvements could include a larger dataset, a web-based prediction interface (Streamlit/Flask), and longitudinal tracking over a semester.

---

##  Author

**Aanshik Mangla **  
B.Tech / CSE core 
Course: Data Science & Machine Learning  
Submitted on VITyarthi Platform

---

##  License

This project is submitted for academic evaluation purposes. Not licensed for commercial use.
