import numpy as np
import pandas as pd

def calculate_Lin_table(x_values, y_values):
    n = len(x_values)

    x = np.array(x_values)
    y = np.array(y_values)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    beta_0 = (sum_y - beta_1 * sum_x) / n

    xy = x * y
    x2 = x ** 2
    y_hat = beta_0 + beta_1 * x
    y_minus_yhat = np.around((y - y_hat),decimals = 2)
    abs_y_minus_yhat = np.abs(y_minus_yhat)
    y_minus_yhat_sq = np.around((y_minus_yhat ** 2),decimals = 2)

    data = {
        'x': x,
        'y': y,
        'x*y': xy,
        'x^2': x2,
        'y(hat)': y_hat,
        'y - y(hat)': y_minus_yhat,
        '|y - y(hat)|': abs_y_minus_yhat,
        '(y - y(hat))^2': y_minus_yhat_sq
    }
    
    df = pd.DataFrame(data)

    # Calculate error metrics
    MAE = np.mean(abs_y_minus_yhat)
    MSE = np.mean(y_minus_yhat_sq)
    RMSE = np.sqrt(MSE)
    
    return df, beta_0, beta_1, MAE, MSE, RMSE

x_values = [1,2,3,4,5]
y_values = [1.2,1.8,2.6,3.2,3.8]

# Now we unpack all returned values from the function
table, beta_0, beta_1, MAE, MSE, RMSE = calculate_Lin_table(x_values, y_values)

print("Beta Naught (β₀):", beta_0)
print("Beta 1 (β₁):", beta_1)
print("\nTable:")
print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

print("\nError Metrics:")
print(f"Mean Absolute Error (MAE): {MAE:.2f}")
print(f"Mean Squared Error (MSE): {MSE:.2f}")
print(f"Root Mean Squared Error (RMSE): {RMSE:.2f}")