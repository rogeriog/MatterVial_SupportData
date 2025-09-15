import glob
import re
import os

# Define the directory where your files are located
# YOU MUST REPLACE THIS PATH with the actual path on your system
path_to_files = './'
file_pattern = os.path.join(path_to_files, '*.out')

# Use a set to store unique scores from all files
unique_scores_set = set()

# Loop through all files matching the pattern
for filename in glob.glob(file_pattern):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            # Find all scores in the file using a regular expression
            scores_str = re.findall(r'Final score =\s*([\d.]+)', content)
            
            # Add each score to the set. Duplicates will be ignored automatically.
            for score in scores_str:
                unique_scores_set.add(float(score))
                
    except IOError as e:
        print(f"Error reading file {filename}: {e}")

# Check if any unique scores were found
if not unique_scores_set:
    print("No unique scores were found. Please check the file path and pattern.")
else:
    # Convert the set to a list for calculations
    unique_scores_list = list(unique_scores_set)
    
    # --- Calculate Average ---
    mean_score = sum(unique_scores_list) / len(unique_scores_list)

    # --- Calculate Average Deviation ---
    average_deviation = sum(abs(score - mean_score) for score in unique_scores_list) / len(unique_scores_list)

    # Print the results
    print(f"Total unique scores found: {len(unique_scores_list)}")
    print(f"Average of the unique scores: {mean_score}")
    print(f"Average deviation of the unique scores: {average_deviation}")