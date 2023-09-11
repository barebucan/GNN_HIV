import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_balance_split_and_save_dataset(csv_file, train_csv, val_csv, test_size=0.2, random_state=None):
    # Load the CSV file into a DataFrame
    dataset = pd.read_csv(csv_file)

    # Split the dataset into label 0 and label 1 examples
    label_0_data = dataset[dataset['HIV_active'] == 0]
    label_1_data = dataset[dataset['HIV_active'] == 1]

    # Split label 0 data into training and validation sets
    label_0_train_data, label_0_val_data = train_test_split(label_0_data, test_size=test_size, random_state=random_state)

    # Split label 1 data into training and validation sets
    label_1_train_data, label_1_val_data = train_test_split(label_1_data, test_size=test_size, random_state=random_state)

    # Duplicate label 1 examples for the training set
    label_1_train_data_duplicated = label_1_train_data.sample(n=len(label_0_train_data), replace=True)

    # Concatenate label 0 and duplicated label 1 training data
    train_data = pd.concat([label_0_train_data, label_1_train_data_duplicated])

    # Concatenate label 0 validation data with non-duplicated label 1 validation data
    val_data = pd.concat([label_0_val_data, label_1_val_data])

    # Save training and validation sets to separate CSV files
    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)

# Example usage
csv_file = r'data\raw\HIV.csv'
train_csv = r'data\raw\HIV_train.csv'
val_csv = r'data\raw\HIV_val.csv'
test_size = 0.8
random_state = 42

load_balance_split_and_save_dataset(csv_file, train_csv, val_csv, test_size, random_state)
