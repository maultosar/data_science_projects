import os
import pandas as pd


def parse_krakow_pollution_data(directory_path, output_path):
    aggregated_data = []

    # Function to determine the header row and data start row
    def get_header_and_data_start_rows(df):
        header_row = 0
        if not pd.isna(pd.to_numeric(df.iloc[0, 1], errors='coerce')):
            header_row = 1

        # Finding the data start row
        data_start_row = header_row + 1
        while data_start_row < len(df) and pd.isna(
                pd.to_numeric(df.iloc[data_start_row, 1], errors='coerce')):  # Check the second column
            data_start_row += 1

        if data_start_row >= len(df):
            raise ValueError(
                "Data start row not found within the first 10 rows. Consider increasing the initial data read length.")

        return header_row, data_start_row

    all_files = os.listdir(directory_path)
    total_files = len(all_files)

    # Loop through each file in the directory
    for idx, file_name in enumerate(all_files, start=1):
        print(f"Processing file {idx} of {total_files}: {file_name}...")

        file_path = os.path.join(directory_path, file_name)

        # Load initial data to determine the header row and data start row
        initial_data = pd.read_excel(file_path, header=None, nrows=10,
                                     engine='openpyxl')  # Assuming header is within the first 10 rows
        initial_data = initial_data.fillna(0)
        initial_data = initial_data.replace({',': '.'}, regex=True)
        header_row, data_start_row = get_header_and_data_start_rows(initial_data)

        # Load the actual data using the determined header row and skiprows
        data = pd.read_excel(file_path, header=header_row, skiprows=range(header_row + 1, data_start_row),
                             engine='openpyxl')
        data = data.replace({',': '.'}, regex=True)

        # Identify the pollutant based on the file name
        pollutant = 'CO' if 'CO' in file_name else 'NO2'

        # Get the numeric index positions for the station columns
        station_column_names = ["MpKrakowWIOSAKra6117", "MpKrakowWIOSBulw6118", "MpKrakAlKras", "MpKrakBulwar"]
        first_col_index = next((data.columns.get_loc(col) for col in data.columns if col in station_column_names), None)
        station_column_indices = [first_col_index] if first_col_index else []

        # Extract the relevant data using .iloc without filling NaN values
        df = data.iloc[:, [0] + station_column_indices].copy()
        df.columns = ['Datetime'] + [pollutant for _ in station_column_indices]

        # Convert 'Datetime' to datetime format
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

        # Filter out rows where 'Datetime' is NaT (Not a Timestamp)
        df = df.dropna(subset=['Datetime'])

        # Set 'Datetime' as index and resample to aggregate data to monthly sum
        df.set_index('Datetime', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Explicit conversion to numeric

        # Debugging: Checking dtypes before resampling
        print("Dtypes before resampling:\n", df.dtypes)

        try:
            monthly_data = df.resample('MS').mean()
            aggregated_data.append(monthly_data)
        except Exception as e:
            print(f"Error occurred while processing {file_name}: {str(e)}")
            print(df.head())  # Print the top rows of the dataframe for inspection

    final_data = pd.concat(aggregated_data).groupby('Datetime').mean()
    final_data = final_data.fillna(0)
    final_data = final_data.sort_index()
    final_data.to_csv(output_path)

    print("Aggregation complete!")


parse_krakow_pollution_data("c:/path/measurements_folder_with_exported_co_no2_files", "output_data.csv")
