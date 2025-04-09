# These are some pre-packaged utility functions used during model training to minimize code duplication.

import yaml
import matplotlib.pylab as plt
import os
import re

# Load training configuration from a YAML file
def load_config(config_path):
    """"Parse model training settings from a .yml config file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Plot Visualization
# Accepts variable arguments for configuration
def plot_img(xlabel_name, ylabel_name, title_name, file_name, *args):
    # 1. Set figure size
    plt.figure(figsize=(12,5))
    # 2. Configure axis labels
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.title(title_name)
    # 3. Plot curves
    
    for i in args:
        plt.plot(i[0], label=i[1], color=i[2])   # [list_a, "label", "red"]
    plt.legend()   # Display legend (if applicable)
    # 4. Save figure
    plt.savefig(file_name)
    plt.tight_layout()  # Auto-adjust layout for dynamic scaling
    plt.close()


# Saves training metadata to a text file
def save_txt(save_list: list, file_name: str):
    with open(file_name, "w") as f:
        for i in save_list:
            f.write(str(i)+", ")
        f.write("\n")


# Retrieve the most recently trained model
def get_largest_pth_file(folder_path, file_name):
    # Get all filenames in the directory
    files = os.listdir(folder_path)
    # Filter files with .pth extension
    pth_files = [f for f in files if f.endswith(".pth") and f.startswith(file_name)]

    # Extract numbers from filenames using re
    max_file = None
    max_number = -1
    for file in pth_files:
        # Match numeric parts in filenames
        match = re.search(r'_(\d+).pth', file)
        if match:
            number = int(match.group(1))
            # print(number)
            # Find the file with the highest number
            if number > max_number:
                max_number = number
                max_file = file

    # Return the full path of the latest file
    return os.path.join(folder_path, max_file) if max_file else None



        