import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from data_utils import load_data, preprocess_input, predict_age_progression
from diabetes_model import DiabetesModel
import pickle
import os

# Remove existing model file to force retraining
# This will ensure the model is trained with the correct features
if os.path.exists("diabetes_model.pkl"):
    os.remove("diabetes_model.pkl")

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("Diabetes Risk Prediction Tool")
st.markdown("""
This application helps predict the risk of diabetes based on several health metrics.
Enter your information below to get a prediction based on machine learning analysis.
""")

# Initialize the model
@st.cache_resource
def get_model():
    model_file = "diabetes_model.pkl"
    
    # Check if model exists, if not train a new one
    if os.path.exists(model_file):
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            # If the model features don't match our current feature set, retrain
            if model.feature_names != ['Pregnancies', 'Glucose', 'SkinThickness', 
                                     'Insulin', 'Age', 'Height', 'Weight',
                                     'FatherDiabetes', 'MotherDiabetes']:
                os.remove(model_file)
                raise FileNotFoundError("Model features don't match current feature set")
        except:
            # If there's any issue loading the model, recreate it
            model = DiabetesModel()
            model.train()
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
    else:
        model = DiabetesModel()
        model.train()
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    
    return model

# Load the model
diabetes_model = get_model()

# Display dataset information
with st.expander("About the Dataset"):
    st.markdown("""
    This app uses the **Modified Pima Indians Diabetes Database** for training the machine learning model. 
    The dataset consists of medical predictor variables such as:
    - Number of pregnancies
    - Glucose plasma concentration
    - Skin thickness
    - Insulin level
    - Height and Weight
    - Family history of diabetes (father and mother)
    - Age
    
    The target variable is a binary outcome of whether the patient has diabetes or not.
    """)
    
    # Load and display sample data
    data = load_data()
    st.write("Sample data from the dataset:")
    st.write(data.head())
    
    # Dataset statistics
    st.write("Dataset statistics:")
    st.write(data.describe())

# Create sidebar for inputs
st.sidebar.header("Enter Your Health Metrics")

# Data input form
with st.sidebar.form("prediction_form"):
    st.subheader("Health Parameters")
    
    # Input fields with reasonable defaults and constraints
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1,
                              help="Blood glucose level after 2 hours in an oral glucose tolerance test")
    
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1,
                                    help="Triceps skin fold thickness")
    
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=80, step=1,
                              help="2-Hour serum insulin")
    
    # Replace BMI with Height and Weight
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.65, step=0.01,
                           help="Height in meters")
    
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1,
                            help="Weight in kilograms")
    
    # Replace diabetes pedigree with father and mother diabetes history
    father_diabetes = st.checkbox("Father has diabetes", value=False,
                                 help="Check if your father has been diagnosed with diabetes")
    
    mother_diabetes = st.checkbox("Mother has diabetes", value=False,
                                 help="Check if your mother has been diagnosed with diabetes")
    
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
    
    # Model selection
    model_type = st.selectbox(
        "Select ML Model",
        options=["Random Forest", "Logistic Regression"]
    )
    
    # Add age progression checkbox
    show_age_progression = st.checkbox("Show diabetes risk progression with age", value=True,
                                    help="See how your diabetes risk changes as you age")
    
    submit_button = st.form_submit_button("Predict Diabetes Risk")

# Main content area for results
if submit_button:
    # Convert boolean values to integers for model input
    father_diabetes_int = 1 if father_diabetes else 0
    mother_diabetes_int = 1 if mother_diabetes else 0
    
    # Create input data frame with the new features
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'FatherDiabetes': [father_diabetes_int],
        'MotherDiabetes': [mother_diabetes_int]
    })
    
    # Preprocess the input
    processed_data = preprocess_input(input_data)
    
    # Make prediction
    diabetes_model.set_model_type(model_type)
    prediction, probability = diabetes_model.predict(processed_data)
    
    # Display results
    st.header("Prediction Results")
    
    # Create two columns for results and visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Display prediction result
        if prediction[0] == 1:
            st.error("**Result: High diabetes risk detected**")
        else:
            st.success("**Result: Low diabetes risk detected**")
        
        # Display probability
        st.metric(
            label="Probability of Diabetes",
            value=f"{probability[0][1]:.2%}"
        )
        
        # Display model information
        st.info(f"Model used: {model_type}")
        
        # Model accuracy
        accuracy = diabetes_model.get_accuracy()
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        # Age progression section
        if show_age_progression:
            st.subheader("Diabetes Risk Progression with Age")
            
            try:
                # Calculate risk progression
                age_progression_data = predict_age_progression(diabetes_model, input_data)
                
                # Create line plot for age progression
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(x='Age', y='Risk Probability', data=age_progression_data, 
                            marker='o', linewidth=2, ax=ax)
                
                # Add vertical line at current age
                ax.axvline(x=age, color='red', linestyle='--', 
                         label=f'Current Age: {age}')
                
                # Highlight age 30
                ax.axvline(x=30, color='green', linestyle=':', 
                         label='Age 30')
                
                ax.set_title('Diabetes Risk Progression with Age')
                ax.set_ylabel('Risk Probability (%)')
                ax.set_ylim(0, 1)
                ax.set_xlim(min(age_progression_data['Age']), max(age_progression_data['Age']))
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format y-axis as percentage
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
                
                st.pyplot(fig)
                
                # Find risk at age 30
                risk_at_30 = float(age_progression_data[age_progression_data['Age'] == 30]['Risk Probability'].iloc[0])
                st.metric("Diabetes Risk at Age 30", f"{risk_at_30:.2%}")
            except Exception as e:
                st.warning(f"Could not generate age progression chart: {str(e)}")
        
    with col2:
        # Display feature importance if Random Forest is selected
        if model_type == "Random Forest":
            st.subheader("Feature Importance")
            try:
                importances = diabetes_model.get_feature_importance()
                
                # If lengths don't match, adjust feature names to match importances length
                feature_list = diabetes_model.feature_names.copy()
                if len(feature_list) != len(importances):
                    # Ensure feature list is the same length as importances
                    if len(feature_list) < len(importances):
                        # Add "Unknown Feature" for any extra importances
                        feature_list = feature_list + [f"Unknown Feature {i+1}" for i in range(len(importances) - len(feature_list))]
                    else:
                        # Trim feature list if it's too long
                        feature_list = feature_list[:len(importances)]
                    
                    # Show warning but continue
                    st.warning(f"Feature names ({len(diabetes_model.feature_names)}) and importance values ({len(importances)}) have different lengths. Adjusted visualization accordingly.")
                
                # Create feature importance plot
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_df = pd.DataFrame({
                    'Feature': feature_list,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis', ax=ax)
                ax.set_title('Feature Importance for Diabetes Prediction')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display feature importance: {str(e)}")
        else:
            # Display coefficients for Logistic Regression
            st.subheader("Feature Coefficients")
            try:
                coefficients = diabetes_model.get_feature_importance()
                
                # If lengths don't match, adjust feature names to match coefficients length
                feature_list = diabetes_model.feature_names.copy()
                if len(feature_list) != len(coefficients):
                    # Ensure feature list is the same length as coefficients
                    if len(feature_list) < len(coefficients):
                        # Add "Unknown Feature" for any extra coefficients
                        feature_list = feature_list + [f"Unknown Feature {i+1}" for i in range(len(coefficients) - len(feature_list))]
                    else:
                        # Trim feature list if it's too long
                        feature_list = feature_list[:len(coefficients)]
                    
                    # Show warning but continue
                    st.warning(f"Feature names ({len(diabetes_model.feature_names)}) and coefficient values ({len(coefficients)}) have different lengths. Adjusted visualization accordingly.")
                
                # Create coefficient plot
                fig, ax = plt.subplots(figsize=(10, 6))
                coef_df = pd.DataFrame({
                    'Feature': feature_list,
                    'Coefficient': coefficients
                }).sort_values('Coefficient', ascending=False)
                
                sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis', ax=ax)
                ax.set_title('Feature Coefficients for Diabetes Prediction')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display feature coefficients: {str(e)}")
    
    # Additional visualizations of input data compared to population
    st.header("Your Metrics Compared to Population")
    
    try:
        # Get population data
        population_data = diabetes_model.get_training_data()
        
        # Select a subset of features to visualize
        features_to_plot = ['Glucose', 'Height', 'Weight', 'Age']
        
        # Only use features that exist in both population_data and input_data
        valid_features = [f for f in features_to_plot if f in population_data.columns]
        
        if valid_features:
            # Create comparison plots - adjust number of subplots based on valid features
            n_features = len(valid_features)
            n_cols = min(2, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            
            # Convert axes to array if it's not already
            if n_features == 1:
                axes = np.array([axes])
            elif n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            
            # Plot each feature
            for i, feature in enumerate(valid_features):
                if n_features > 1:
                    ax = axes.flatten()[i]
                else:
                    ax = axes
                
                # Plot histogram of population data
                sns.histplot(x=population_data[feature], ax=ax, kde=True, color='skyblue', label='Population')
                
                # Plot vertical line for user input
                user_value = float(input_data[feature].iloc[0])
                ax.axvline(x=user_value, color='red', linestyle='--', linewidth=2, 
                          label=f'Your {feature}: {user_value:.1f}')
                
                ax.set_title(f'Your {feature} vs Population')
                ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No valid features found to compare with population.")
    except Exception as e:
        st.error(f"Error generating population comparison: {str(e)}")
    
    # Explanation of prediction
    st.header("Understanding Your Prediction")
    
    try:
        # Get top factors based on feature importance and user values
        if model_type == "Random Forest":
            importances = diabetes_model.get_feature_importance()
        else:
            importances = np.abs(diabetes_model.get_feature_importance())
        
        # If lengths don't match, adjust feature names
        feature_list = diabetes_model.feature_names.copy()
        if len(feature_list) != len(importances):
            # Ensure feature list is the same length as importances
            if len(feature_list) < len(importances):
                feature_list = feature_list + [f"Unknown Feature {i+1}" for i in range(len(importances) - len(feature_list))]
            else:
                feature_list = feature_list[:len(importances)]
        
        # Get user values, extending if necessary
        user_values = list(input_data.iloc[0].values)
        if len(user_values) < len(feature_list):
            user_values = user_values + [0] * (len(feature_list) - len(user_values))
        else:
            user_values = user_values[:len(feature_list)]
        
        # Create a dataframe with user values and importances
        explanation_df = pd.DataFrame({
            'Feature': feature_list,
            'Your Value': user_values,
            'Importance': importances
        })
        
        # Add population average for comparison, handling any missing features
        pop_avgs = []
        for feat in feature_list:
            if feat in population_data.columns:
                pop_avgs.append(population_data[feat].mean())
            else:
                pop_avgs.append(0)  # Default value for missing features
        
        explanation_df['Population Average'] = pop_avgs
        
        # Calculate percentage difference from population mean (safely)
        pct_diffs = []
        for i, row in explanation_df.iterrows():
            if row['Population Average'] != 0:
                pct_diff = ((row['Your Value'] - row['Population Average']) / row['Population Average'] * 100)
            else:
                pct_diff = 0
            pct_diffs.append(pct_diff)
        
        explanation_df['% Difference'] = pct_diffs
        
        # Sort by importance
        explanation_df = explanation_df.sort_values('Importance', ascending=False)
        
        # Display the top contributing factors
        st.subheader("Top Contributing Factors to Your Prediction")
        st.write(explanation_df)
    except Exception as e:
        st.warning(f"Could not analyze feature contributions: {str(e)}")
    
    # Health recommendations
    st.header("Health Recommendations")
    
    if prediction[0] == 1:
        st.warning("""
        Based on your results, here are some general recommendations:
        
        1. **Consult with a healthcare provider** to discuss these results and get personalized advice.
        2. **Monitor your blood glucose levels** regularly.
        3. **Maintain a healthy diet** low in sugar and refined carbohydrates.
        4. **Engage in regular physical activity** - aim for at least 150 minutes of moderate exercise per week.
        5. **Maintain a healthy weight** - even a modest weight loss can improve insulin sensitivity.
        
        Remember: This is a screening tool and not a medical diagnosis. Only healthcare professionals can diagnose diabetes.
        """)
    else:
        st.info("""
        Based on your results, here are some general recommendations to maintain good health:
        
        1. **Continue regular health check-ups** with your healthcare provider.
        2. **Maintain a balanced diet** rich in vegetables, fruits, whole grains, and lean proteins.
        3. **Stay physically active** with regular exercise.
        4. **Manage stress** through relaxation techniques and adequate sleep.
        5. **Limit alcohol consumption** and avoid smoking.
        
        Remember: This is a screening tool and not a medical diagnosis. Continue to follow your healthcare provider's advice.
        """)

# Additional information section
st.header("About Diabetes")
with st.expander("What is Diabetes?"):
    st.markdown("""
    **Diabetes mellitus** is a chronic metabolic disorder characterized by elevated blood glucose levels (hyperglycemia) 
    resulting from defects in insulin secretion, insulin action, or both.
    
    There are three main types of diabetes:
    
    1. **Type 1 Diabetes**: The body fails to produce insulin. People with type 1 diabetes require daily insulin injections.
    
    2. **Type 2 Diabetes**: The body doesn't produce enough insulin or cells become resistant to insulin. This type accounts for 90-95% of diabetes cases.
    
    3. **Gestational Diabetes**: Occurs during pregnancy and typically resolves after delivery, although it increases the risk of type 2 diabetes later in life.
    
    Early detection and treatment of diabetes can decrease the risk of developing complications.
    """)

with st.expander("Risk Factors for Diabetes"):
    st.markdown("""
    Several factors can increase the risk of developing diabetes:
    
    - Family history of diabetes
    - Overweight or obesity
    - Physical inactivity
    - Race/ethnicity (higher risk in African Americans, Hispanic/Latino Americans, American Indians, Pacific Islanders)
    - Age (risk increases with age, especially after 45)
    - High blood pressure
    - Abnormal cholesterol levels
    - History of gestational diabetes
    - Polycystic ovary syndrome
    
    Many of these factors are captured in the prediction model used by this application.
    """)

# Footer
st.markdown("---")
st.caption("This application is for educational purposes only and should not replace professional medical advice.")
