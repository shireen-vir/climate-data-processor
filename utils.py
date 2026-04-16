import pandas as pd
import numpy as np

"""
Module for climate data processing utilities.

This module provides functions for handling and manipulating climate-related data.
It includes data cleaning, filtering, and transformation methods.
"""

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")

def filter_data(data, column, value):
    filtered_data = data[data[column] == value]
    return filtered_data

def transform_data(data, column, func):
    data[column] = data[column].apply(func)
    return data

def main():
    file_path = 'climate_data.csv'
    data = load_data(file_path)
    filtered_data = filter_data(data, 'region', 'North')
    transformed_data = transform_data(filtered_data, 'temperature', np.square)
    print(transformed_data.head())

if __name__ == "__main__":
    main()