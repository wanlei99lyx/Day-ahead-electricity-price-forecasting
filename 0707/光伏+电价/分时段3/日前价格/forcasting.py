import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_price_comparison(true_data, pred_data, days, points_per_day):
    """
    Plots and saves the comparison chart of actual vs. predicted prices.
    """
    total_points = len(true_data)
    print(f"\n--- Generating {days}-Day Price Actual vs. Predicted Chart... ---")
    plt.figure(figsize=(18, 9))

    # Use the data's index (0, 1, 2, ...) for the x-axis
    plt.plot(true_data.index, true_data.values, label='Actual Price', linestyle='-', marker='o',
             markersize=4,
             color='blue', alpha=0.8)
    plt.plot(pred_data.index, pred_data.values, label='Predicted Price', linestyle='--', marker='x',
             markersize=4,
             color='red', alpha=0.8)

    plt.title(f'{days}-Day Price Prediction vs Actual', fontsize=18)
    plt.xlabel(f'Data Points (Total {total_points} points over {days} days)', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set ticks based on data point positions
    tick_positions = np.arange(0, total_points, points_per_day)
    tick_labels = [f'Day {i + 1}' for i in range(days)]

    # Ensure labels and positions match
    if len(tick_labels) > len(tick_positions):
        tick_labels = tick_labels[:len(tick_positions)]

    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=30, ha="right")

    plt.tight_layout()
    save_path = f'price_prediction_vs_actual_{days}_days.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Price comparison chart saved to file: {save_path}")


def plot_daily_mape(metrics_data, days):
    """
    Plots and saves the daily MAPE trend chart.
    """
    print(f"\n--- Generating {days}-Day Daily MAPE Curve Chart... ---")
    plt.figure(figsize=(15, 7))
    plt.plot(metrics_data.index, metrics_data['MAPE'], marker='o', linestyle='-', color='purple', label='Daily MAPE')
    plt.title(f'Daily MAPE Over {days} Days', fontsize=16)
    plt.xlabel('Day', fontsize=14)
    plt.ylabel('MAPE (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    save_path = f'daily_mape_curve_{days}_days.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Daily MAPE curve chart saved to file: {save_path}")


def main():
    """
    Main function to execute the data merging, calculation, and visualization process.
    """
    # --- 1. Load your data from files ---
    print("--- Loading data files... ---")
    try:
        df1 = pd.read_csv('1-10/日前_final_prediction_vs_actual_1-10_k20_30天.csv')
        df2 = pd.read_csv('11-13/日前_final_prediction_vs_actual_11-13_k20_30天.csv')
        df3 = pd.read_csv('14-24/日前_final_prediction_vs_actual_14-24_k25_30天.csv')

        # Verify that the required columns exist
        required_columns = ['Actual_Price', 'Predicted_Price']
        for df_name, df in [('df1', df1), ('df2', df2), ('df3', df3)]:
            if not all(col in df.columns for col in required_columns):
                print(
                    f"Error: File {df_name} is missing 'Actual_Price' or 'Predicted_Price' column. Please check your files.")
                return

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure the file paths and names are correct.")
        return

    print("Data loading complete:")
    print("Table 1 (1-10h) rows:", len(df1))
    print("Table 2 (11-13h) rows:", len(df2))
    print("Table 3 (14-24h) rows:", len(df3))

    # --- 2. Merge data by day ---
    print("\n--- Merging data tables by day... ---")
    num_days = 30
    points_per_day_d1 = 10
    points_per_day_d2 = 3
    points_per_day_d3 = 11

    # Validate data row count against the number of days
    if not (len(df1) == num_days * points_per_day_d1 and
            len(df2) == num_days * points_per_day_d2 and
            len(df3) == num_days * points_per_day_d3):
        print("Error: One or more data files do not have a row count that matches the 30-day data volume.")
        return

    daily_dfs = []
    for i in range(num_days):
        day_df1 = df1.iloc[i * points_per_day_d1: (i + 1) * points_per_day_d1]
        day_df2 = df2.iloc[i * points_per_day_d2: (i + 1) * points_per_day_d2]
        day_df3 = df3.iloc[i * points_per_day_d3: (i + 1) * points_per_day_d3]

        day_full = pd.concat([day_df1, day_df2, day_df3], ignore_index=True)
        daily_dfs.append(day_full)

    full_period_df = pd.concat(daily_dfs, ignore_index=True)
    print("Data merging complete. Total data points:", len(full_period_df))

    # --- 3. Prepare data and call plotting functions ---
    true_data_series = full_period_df['Actual_Price']
    pred_data_series = full_period_df['Predicted_Price']
    days = 30
    points_per_day = 24  # 10 + 3 + 11

    # Call the price comparison plot function
    plot_price_comparison(true_data_series, pred_data_series, days, points_per_day)

    # --- 4. Calculate MAPE day by day ---
    print("\n--- Calculating MAPE day by day... ---")
    daily_mapes = []
    for i in range(days):
        start_idx = i * points_per_day
        end_idx = (i + 1) * points_per_day

        daily_true = true_data_series[start_idx:end_idx]
        daily_pred = pred_data_series[start_idx:end_idx]

        # Filter out data points where the actual value is 0 to avoid division by zero errors
        non_zero_mask = daily_true != 0
        if not non_zero_mask.all():
            print(f"Note: Day {i + 1} contains actual prices of 0, which have been ignored in the MAPE calculation.")

        mape = np.mean(
            np.abs((daily_true[non_zero_mask] - daily_pred[non_zero_mask]) / daily_true[non_zero_mask])) * 100
        daily_mapes.append(mape)

    # Create a DataFrame for plotting the daily MAPE chart
    metrics_data = pd.DataFrame({
        'MAPE': daily_mapes
    }, index=[f'Day {i + 1}' for i in range(days)])

    # Call the daily MAPE curve plot function
    plot_daily_mape(metrics_data, days)

    # --- 5. Calculate and output the total average MAPE ---
    total_average_mape = metrics_data['MAPE'].mean()
    print("\n--- 30-Day Analysis Results ---")
    print(f"Total Average MAPE: {total_average_mape:.2f}%")


if __name__ == '__main__':
    main()
