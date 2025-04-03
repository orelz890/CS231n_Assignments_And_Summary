
import os
import csv

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the Python file you want to run
folder_path = os.path.join(current_directory, "Rice_Image_Dataset")
csv_file = "Rice_Image_Dataset/image_names.csv"



labels = {"Arborio": 0, "Basmati": 1, "Ipsala": 2, "Jasmine": 3, "Karacadag": 4}

# Open the CSV file in write mode
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file, delimiter='\n', escapechar='\t', quoting=csv.QUOTE_NONE)
    writer.writerow(["image,label"])

    # Traverse the directories
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            rice_folder_path = os.path.join(folder_path, dir_name)
            for oot, dirs, files in os.walk(rice_folder_path):
                for file_name in files:
                    # print(file_name)
                    # Write the image name to the CSV file
                    writer.writerow([f"{dir_name}/{file_name},{labels[dir_name]}"])

print("CSV file created successfully.")
