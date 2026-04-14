# 🛡️ CredSight: Alternative Credit Scoring for the Unbanked

**CredSight** is a machine learning-powered application designed to evaluate creditworthiness for individuals who lack traditional banking history (CIBIL/FICO scores). By leveraging behavioral proxies and alternative data points, the model provides a "Reliability Score" to help financial institutions lend to "New-to-Credit" customers fairly and transparently.

---

## 🚀 The Core Strategy
Traditional credit models often exclude people without bank accounts. **CredSight** ignores demographic biases (like sex or foreign worker status) and focuses on:
* **Stability Proxies:** Residence duration and housing status.
* **Economic Resilience:** Employment history and savings/bonds.
* **Financial Burden:** A custom-engineered "Burden to Stability" ratio comparing loan size to job/living stability.

## 🛠️ Tech Stack
* **Modeling:** XGBoost (Extreme Gradient Boosting)
* **Explainability:** SHAP (Shapley Additive Explanations)
* **Frontend:** Streamlit
* **Data Handling:** Pandas, Scikit-Learn, Joblib
* **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)

---

## 📊 Model Performance & Fairness
The model is trained on the **UCI German Credit Dataset**. 

### Key Features:
1.  **Bias Mitigation:** Explicitly dropped protected attributes (Gender, Nationality) to ensure ethical lending.
2.  **Imbalance Handling:** Used SMOTE to ensure the model learns to identify "High Risk" individuals despite them being a minority in the dataset.
3.  **Explainability:** Every prediction includes a SHAP Force Plot, explaining *why* a user received their specific score—no "Black Box" decisions.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/credsight.git](https://github.com/your-username/credsight.git)
cd credsight
```
2. Install dependencies
```Bash
pip install -r requirements.txt
```
3. Train the Model
Run the notebook or script to generate the trained model and encoders:

```Bash
python trainmodel.py
```
4. Run the App
Launch the Streamlit interface:

```Bash
streamlit run app.py
```
📁 Project Structure
trainmodel.ipynb: Data cleaning, feature engineering (amount per month, burden ratio), SMOTE application, and XGBoost training.

app.py: The Streamlit dashboard featuring a Gauge Chart (300-900 score) and SHAP integration.

credit_model.pkl: The saved XGBoost classifier.

encoders.pkl & label_maps.pkl: Stored transformations to handle categorical alternative data.

💡 How it Works
Input: User enters "proxy" data (e.g., how long they've lived at their current home, their job level, their savings).

Processing: The app calculates the Burden-to-Stability index.

Prediction: XGBoost calculates the probability of risk.

Transformation: The risk probability is scaled into a credit score between 300 and 900.

Transparency: SHAP breaks down the contribution of every input, showing the user exactly what they can improve to get a better score.

⚖️ Disclaimer
This project is a prototype for educational purposes and uses the UCI German Credit dataset. It should not be used as the sole basis for real-world financial lending without further validation and compliance with local financial regulations.


---
