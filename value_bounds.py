import pandas as pd

def modify_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Define the columns and rows to be modified
    columns_to_modify = df.columns[1:10]  # B to J corresponds to column indices 1 to 9 (0-based index)
    rows_to_modify = df.index[1:22597]    # 2 to 22597 corresponds to row indices 1 to 22596 (0-based index)

    # Apply the modifications
    df.loc[rows_to_modify, columns_to_modify] = df.loc[rows_to_modify, columns_to_modify].applymap(
        lambda x: -1 if x < -1 else (1 if x > 1 else x)
    )

    # Save the modified dataframe to the same CSV file
    df.to_csv(file_path, index=False)

# Usage example
file_path = 'solution_submission_alex.csv'
modify_csv(file_path)
