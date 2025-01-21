import pandas as pd
import os

def convert_to_principles():
    directory = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/principle_test_cases/'
    principles_df = pd.DataFrame()
    for filename in os.listdir(directory):
        values = []
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Warning: {file_path} is empty.")
            except pd.errors.EmptyDataError:
                print(f"Error: {file_path} is empty or not a valid CSV.")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
            df.drop(columns=['country'], inplace=True)
            for _, row in df.iterrows():
                value = 0 
                sum = row['rel'] + row['nonrel']
                egal_support = row['rel'] / sum
                value = abs(1 + 9 * egal_support)
                value = round(value, 1)
                values.append(value)
        print(values)
        principles_df[filename] = values

    return principles_df

def generate_means():
    directory = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/principle_test_cases/'
    means_df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Warning: {file_path} is empty.")
            except pd.errors.EmptyDataError:
                print(f"Error: {file_path} is empty or not a valid CSV.")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
            df.drop(columns=['country'], inplace=True)
            means = df.mean()
            means_df[filename] = means

    return means_df

if __name__ == "__main__":
    means_df = generate_means()
    means_df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/means.csv', index=False)
    principles_df = convert_to_principles()
    principles_df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/principles_for_slm.csv', index=False)