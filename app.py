import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components
import plotly.graph_objects as go

# --- 1. LOAD THE ASSETS ---
@st.cache_resource # This keeps the model in memory so the app stays fast
def load_assets():
    model = joblib.load('credit_model.pkl')
    encoders = joblib.load('encoders.pkl')
    columns = joblib.load('model_columns.pkl')
    label_maps = joblib.load('label_maps.pkl')
    explainer = shap.TreeExplainer(model)
    return model, encoders, columns, label_maps, explainer

model, encoders, columns, label_maps, explainer = load_assets()

def friendly_selectbox(label, column_name):
    display_values = list(label_maps[column_name].values())
    selected_display = st.selectbox(label, display_values)

    reverse_map = {v: k for k, v in label_maps[column_name].items()}
    return reverse_map[selected_display]

# Function to render SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def credit_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Credit Score"},
        gauge={
            'axis': {'range': [300, 900]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [300, 500], 'color': "#ff4d4d"},
                {'range': [500, 700], 'color': "#ffd11a"},
                {'range': [700, 900], 'color': "#66cc66"}
            ],
        }
    ))

    fig.update_layout(height=300)
    return fig

def generate_shap_explanation(shap_values, input_df, top_n=3):
    """
    Convert SHAP values into human-readable explanations.
    """
    
    # Create dataframe of contributions
    shap_df = pd.DataFrame({
        "feature": input_df.columns,
        "impact": shap_values
    })

    # Sort impacts
    shap_df = shap_df.sort_values("impact")

    negative = shap_df.head(top_n)   # increases risk
    positive = shap_df.tail(top_n)   # improves score

    return positive, negative

def clean_feature_name(name):
    return name.replace("_", " ").title()

# --- 2. USER INTERFACE ---
st.set_page_config(page_title="CredSight - Unbanked Credit Score", layout="wide")

st.title("🛡️ CredSight: Alternative Credit Scoring")
st.markdown("### For Unbanked & New-to-Credit Customers")
st.write("Enter your details below to calculate your creditworthiness based on behavioral proxies.")

# Create two columns: Left for Input, Right for Result
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("User Information")
    with st.form("input_form"):
        # Numeric Inputs
        age = st.slider("Age in years", 18, 100, 30)
        duration = st.slider("Loan Duration (Months)", 1, 72, 12)
        amount = st.number_input("Requested Credit Amount", 100, 20000, 5000)
        residence = st.slider("Years at Current Residence", 1, 10, 2)
        
        # Categorical Inputs (Using classes from your saved encoders)
        housing = friendly_selectbox("Housing Type", "Housing")
        checking = friendly_selectbox("Checking Account Status","checking account status")
        savings = friendly_selectbox("Savings Account Status","Savings account/bonds")
        purpose = friendly_selectbox("Loan Purpose","Purpose")
        job = friendly_selectbox("Job Category","Job")
        employment = friendly_selectbox("Employment Duration","employment")
        property_type = friendly_selectbox("Property Type","Property")

        submit = st.form_submit_button("Calculate My Score")

# --- 3. PREDICTION & EXPLAINABILITY ---
if submit:
    # A. Prepare the input data (Must match the order of your 'proxies' list)
    # Mapping inputs back to encoded numbers
    input_dict = {
        'checking account status': encoders['checking account status'].transform([checking])[0],
        'Duration in month': duration,
        'Purpose': encoders['Purpose'].transform([purpose])[0],
        'Credit amount': amount,
        'Savings account/bonds': encoders['Savings account/bonds'].transform([savings])[0],
        'employment': encoders['employment'].transform([employment])[0], # Assuming 'A73' or similar; adjust based on your training
        'residence': residence,
        'Property': encoders['Property'].transform([property_type])[0], # Defaulting to 0 for prototype
        'Age in years': age,
        'Housing': encoders['Housing'].transform([housing])[0],
        'Job': encoders['Job'].transform([job])[0]
    }
    
    input_df = pd.DataFrame([input_dict])
    # --- FEATURE ENGINEERING (MUST MATCH TRAINING) ---
    input_df['amount_per_month'] = (input_df['Credit amount'] / input_df['Duration in month'])
    input_df['burden_to_stability'] = input_df['amount_per_month'] / ((input_df['employment'] + 1) * (input_df['residence'] + 1))
    input_df = input_df.reindex(columns=columns, fill_value=0)


    # B. Get Probability and Score
    prob_risk = model.predict_proba(input_df)[0][1]
    threshold = 0.4
    prediction = int(prob_risk > threshold)
    credit_score = int((1 - prob_risk) * 900) # Scale it like a CIBIL/FICO score (300-900)

    with col2:
        st.subheader("Your Credit Result")
        
        st.plotly_chart(credit_score_gauge(credit_score), use_container_width=True)
        if credit_score > 700:
            st.balloons()
            st.success("High Eligibility")
        elif credit_score > 500:
            st.warning("Moderate Eligibility")
        else:
            st.error("High Risk")

        # C. SHAP Force Plot (Individual Explanation)
        st.write("---")
        st.write("#### Why this score?")
        st.write("The chart below shows how your data pushed your score up (blue) or down (red):")
        
        # explainer = shap.TreeExplainer(model)
        # Calculate SHAP values for just this one person
        # shap_vals_single = explainer.shap_values(input_df)
        shap_vals_single = explainer.shap_values(input_df)[0]
        positive, negative = generate_shap_explanation(shap_vals_single, input_df)
        
        # Generate the Force Plot
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_vals_single, 
            input_df,
            link="logit" # Uses logit to show probability impact
        )
        
        st_shap(force_plot, height=150)

        st.write("### 🟢 Factors Improving Your Score")
        for _, row in positive.iterrows():
            st.success(f"{clean_feature_name(row['feature'])} positively influenced your creditworthiness")

        st.write("### 🔴 Factors Increasing Risk")
        for _, row in negative.iterrows():
            st.error(f"{clean_feature_name(row['feature'])} increased your risk level")


st.sidebar.info("This prototype uses Alternative Data proxies to assess creditworthiness for unbanked individuals, bypassing traditional CIBIL scores.")