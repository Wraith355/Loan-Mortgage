import streamlit as st
import numpy as np
import pandas as pd
import os

# Try multiple ways to import joblib
try:
    # Try importing from scikit-learn first (more reliable in some environments)
    from sklearn.externals import joblib
    st.success("Successfully imported joblib from sklearn.externals")
except ImportError:
    try:
        # Try importing standalone joblib
        import joblib
        st.success("Successfully imported standalone joblib")
    except ImportError:
        try:
            # Last resort: try importing from sklearn directly
            from sklearn import joblib
            st.success("Successfully imported joblib from sklearn")
        except ImportError as e:
            st.error(f"""
            Failed to import joblib from any known location. 
            Error: {str(e)}
            
            Current Python path: {os.environ.get('PYTHONPATH', 'Not set')}
            Installed packages:
            """)
            
            # Try to show installed packages
            try:
                import pkg_resources
                installed_packages = [f"{dist.key} {dist.version}" for dist in pkg_resources.working_set]
                st.code("\n".join(installed_packages))
            except:
                st.error("Could not list installed packages")
            
            st.stop()

def main():
    st.set_page_config(
        page_title="Mortgage Delinquency Predictor",
        page_icon="🏠",
        layout="wide"
    )

    st.title("Mortgage Delinquency Predictor")

    # Load model and scaler with detailed error handling
    try:
        model_path = "svc_mortgage_model.pkl"
        scaler_path = "scaler.pkl"
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.stop()
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at: {scaler_path}")
            st.stop()
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Get feature names from the scaler if available
        feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
        
        st.success("✅ Model and scaler loaded successfully!")
    except Exception as e:
        st.error(f"""
        Error loading model files: {str(e)}
        
        Current working directory: {os.getcwd()}
        Files in directory: {os.listdir('.')}
        """)
        st.stop()

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Input fields with more descriptive labels and tooltips
        inputs = {}
        inputs["CreditScore"] = st.slider(
            "Credit Score", 
            300, 850, 700, 
            help="FICO credit score of the borrower"
        )
        inputs["FirstPaymentDate_Year"] = st.number_input(
            "First Payment Year", 
            min_value=2000, 
            max_value=2050, 
            value=2023,
            help="Year when the first mortgage payment is due"
        )
        inputs["FirstPaymentDate_Month"] = st.selectbox(
            "First Payment Month", 
            list(range(1, 13)),
            help="Month when the first mortgage payment is due"
        )
        inputs["MaturityDate_Year"] = st.number_input(
            "Maturity Year", 
            min_value=2000, 
            max_value=2100, 
            value=2053,
            help="Year when the mortgage will be fully paid off"
        )
        inputs["MaturityDate_Month"] = st.selectbox(
            "Maturity Month", 
            list(range(1, 13)),
            help="Month when the mortgage will be fully paid off"
        )

    with col2:
        inputs["LoanTerm"] = st.selectbox(
            "Loan Term (months)", 
            [180, 240, 360],
            help="Duration of the mortgage in months"
        )
        inputs["PropertyType"] = st.selectbox(
            "Property Type", 
            ["SF", "CO", "PU"],
            help="SF=Single Family, CO=Condo, PU=PUD"
        )
        inputs["OccupancyStatus"] = st.selectbox(
            "Occupancy", 
            ["O", "I", "S"],
            help="O=Owner Occupied, I=Investment Property, S=Second Home"
        )
        inputs["LoanPurpose"] = st.selectbox(
            "Purpose", 
            ["P", "C", "N"],
            help="P=Purchase, C=Cash-out Refinance, N=No Cash-out Refinance"
        )
        inputs["LoanAmount"] = st.number_input(
            "Loan Amount", 
            min_value=0,
            value=250000,
            help="Total amount of the mortgage loan"
        )
        inputs["BorrowerIncome"] = st.number_input(
            "Borrower Income", 
            min_value=0,
            value=80000,
            help="Annual income of the borrower"
        )

    if st.button("Predict", type="primary"):
        try:
            # Create DataFrame with input values
            df = pd.DataFrame([inputs])
            
            # Calculate derived features
            df['DTIRatios'] = (df['LoanAmount'] / df['BorrowerIncome']).clip(0, 100)
            df['MonthsToMaturity'] = ((df['MaturityDate_Year'] - df['FirstPaymentDate_Year']) * 12 + 
                                    (df['MaturityDate_Month'] - df['FirstPaymentDate_Month']))
            
            # Add remaining required features with default values
            default_features = {
                'OrigInterestRate': 6.5,  # typical mortgage rate
                'OrigUPB': lambda x: x['LoanAmount'],
                'Channel': 'R',  # Retail
                'SellerName': 'OTHER',
                'ServicerName': 'OTHER',
                'NumBorrowers': 1,
                'Loan2Value': 80,  # typical LTV
                'MI_Pct': 0,
                'BorrowerCreditScore': lambda x: x['CreditScore'],
                'CoBorrowerCreditScore': 0,
                'LoanProductType': 'FRM',
                'RelocationIndicator': 'N',
                'PropertyState': 'XX',
                'PostalCode': '00000',
                'ModFlag': 'N',
                'ProgramIndicator': 'N',
                'MSA': '00000'
            }
            
            for feat, value in default_features.items():
                df[feat] = value if not callable(value) else df.apply(value, axis=1)
            
            # Ensure categorical columns are properly encoded
            categorical_cols = ['PropertyType', 'OccupancyStatus', 'LoanPurpose', 'Channel', 
                            'SellerName', 'ServicerName', 'LoanProductType', 
                            'RelocationIndicator', 'PropertyState', 'ModFlag', 
                            'ProgramIndicator']
            
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category').cat.codes
            
            # Ensure columns match scaler's expected features
            if feature_names is not None:
                missing_cols = set(feature_names) - set(df.columns)
                if missing_cols:
                    for col in missing_cols:
                        df[col] = 0
                df = df[feature_names]
            
            # Scale features
            X = scaler.transform(df)
            
            # Make prediction
            prediction = model.predict(X)
            
            # Display result with confidence
            st.markdown("---")
            if prediction[0] == 0:
                st.success("✅ Prediction: Not Delinquent")
            else:
                st.error("❌ Prediction: Potential Delinquency Risk")
            
            # Add disclaimer
            st.info("Note: This prediction is based on limited information and should not be used as the sole factor in decision making.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
