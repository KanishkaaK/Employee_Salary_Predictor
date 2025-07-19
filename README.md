###  1. **README.md**

````markdown
#  Employee Income Prediction 

This project trains a **Random Forest Classifier** to predict whether an individual earns more than $50K/year, using the UCI Adult Income dataset.

---

##  Files

| File | Description |
|------|-------------|
| `Salary pred.csv` | Cleaned dataset used for training |
| `train_model.py`  | Training script (you posted above) |
| `test_input.py`   | Predicts salary from user input |
| `salary_predictor_model.joblib` | Trained model |
| `scaler.joblib`   | StandardScaler used on numeric features |
| `label_encoders.joblib` | Encoders for all categorical features |

---

##  Model Info

- Model: `RandomForestClassifier`
- Accuracy: ~84.68%
- Preprocessing:
  - Categorical: `LabelEncoder`
  - Numeric: `StandardScaler`
- Saved with: `joblib`

---

##  How to Use

###  1. Train the Model

```bash
python train_model.py
````

This saves 3 files:

* `salary_predictor_model.joblib`
* `scaler.joblib`
* `label_encoders.joblib`

---

###  2. Run Test Input

Edit values in `test_input.py` to test with new inputs.

```bash
python test_input.py
```

Example output:

```
Predicted Income Class: >50K
```

---

##  Dependencies

Install them using:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
scikit-learn
joblib
```

---

##  Dataset

* Source: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
* Target: `income` column (`<=50K` or `>50K`)

---

##  License

MIT License. Free to use and modify.


