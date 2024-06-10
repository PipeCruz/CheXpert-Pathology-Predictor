import csv
import re

def tensor_to_number(value):
    # Extract numerical part using regular expression
    match = re.match(r'tensor\((-?\d+(\.\d+)?([eE][+-]?\d+)?)\)', value)
    if match:
        return float(match.group(1))
    else:
        # Return original value if it doesn't match the expected format
        return value

def convert_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            new_row = []
            for j, value in enumerate(row):
                # Check if the cell is within the specified range
                if 1 <= j <= 9 and 1 <= i <= 25595:
                    new_row.append(tensor_to_number(value))
                else:
                    new_row.append(value)
            writer.writerow(new_row)

# Replace 'input.csv' and 'output_new.csv' with your file paths
convert_csv('solution_submission_alexnet.csv', 'solution_submission_alex.csv')
