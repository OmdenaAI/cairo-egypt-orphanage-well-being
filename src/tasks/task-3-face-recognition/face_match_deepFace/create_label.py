"""
    Author: Ahmed Sobhi
    Department: Data Science
    Created_at: 2023-09-10
    Objective: Generate Label CSV file.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import pandas as pd

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Generate Label.')

# Define command-line arguments with default values
parser.add_argument('--label_path', default='data/label', help='Label directory path (default: /data/label)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
label_path = args.label_path

label_data = []

if __name__ == '__main__':
    # for dir_name in os.listdir(label_path):
    #     print(dir_name)
    for r, _, f in os.walk(label_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    label_data.append([file, file.split('.')[0]])

    # Create dataFrame
    if len(label_data) != 0:
        # Image Label Data Was found
        df_label = pd.DataFrame(label_data, columns=['image_name', 'label'])

        # Save as csv file
        df_label.to_csv(f'{label_path}/label.csv', index=False)

        print(f'Label CSV file was created at {label_path}')
        