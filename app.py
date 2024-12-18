import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit App
def main():
    # Title of the app
    st.title('Linear Regression App')

    # Upload CSV File
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Show dataset preview
        st.write("Dataset preview:")
        st.write(df.head())

        # Let the user select X and Y columns for regression
        all_columns = df.columns.tolist()
        x_feature = st.selectbox('Select the X feature:', all_columns)
        y_feature = st.selectbox('Select the Y feature:', all_columns)

        # Check if X and Y are different
        if x_feature != y_feature:
            # Prepare the data for training
            X = df[[x_feature]]
            y = df[y_feature]

            # Split data into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize the linear regression model
            model = LinearRegression()

            # Train the model
            model.fit(X_train, y_train)

            # Predict using the trained model
            y_pred = model.predict(X_test)

            # Display the coefficients and intercept
            st.write(f'Coefficient: {model.coef_[0]}')
            st.write(f'Intercept: {model.intercept_}')

            # Display Mean Squared Error
            mse = mean_squared_error(y_test, y_pred)
            st.write(f'Mean Squared Error: {mse}')

            # Plotting the results
            plt.figure(figsize=(8, 6))
            plt.scatter(X_test, y_test, color='blue', label='Actual')
            plt.plot(X_test, y_pred, color='red', label='Predicted')
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            plt.title(f'Linear Regression: {x_feature} vs {y_feature}')
            plt.legend()
            st.pyplot(plt)

        else:
            st.error("X and Y features must be different.")

# Run the app
if __name__ == '__main__':
    main()
