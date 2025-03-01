from LinearRegression import calculate_Lin_table
import streamlit as st

st.title("Linear Regression Calculator App")

input1 = st.text_input("Enter X values (use space to separate values):")
input2 = st.text_input("Enter Y values (use space to separate values):")

if st.button("Submit"):
    try:
        l1 = [round(float(num.strip()),2) for num in input1.split()]
        l2 = [round(float(num.strip()),2) for num in input2.split()]
        table, beta_0, beta_1, MAE, MSE, RMSE = calculate_Lin_table(l1,l2)
        st.write("Beta Naught (β₀):", round(beta_0,2))
        st.write("Beta 1 (β₁):", round(beta_1,2))
        st.write("\nError Metrics:")
        st.write(f"Mean Absolute Error (MAE): {MAE:.2f}")
        st.write(f"Mean Squared Error (MSE): {MSE:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {RMSE:.2f}")
        st.write("\nTable:")
        st.write(table)
    except ValueError:
        st.write("Please enter valid numbers separated by commas.")
