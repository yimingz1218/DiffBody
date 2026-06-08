import re
import os
from tqdm import tqdm

# Define the regex patterns to match the desired numbers
tar_file_pattern = r'Process (\d+) .tar files'
total_images_pattern = r'totally contain (\d+) images'
person_images_pattern = r'(\d+) of them contains person'
# ratio_pattern = r'ratio is (\d+\.\d+)'
# peak_memory_pattern = r'Peak GPU memory usage: (\d+\.\d+) MB'
time_cost_pattern = r'The overall time cost is (\d+\.\d+) seconds'
# avg_time_pattern = r'avg each tar file process time is (\d+\.\d+) seconds'

# Create a list to store the extracted numbers
numbers = []
tar_file_num = total_images_num = person_images_num = time_cost_num = 0

file_names = [os.path.join('./logs', x)
                                 for x in sorted(os.listdir('./logs')) if
                                 x.startswith('laion')]

# Iterate over the files
for file_name in tqdm(file_names):
    # Read the contents of the file
    with open(file_name, 'r') as file:
        contents = file.read()
    
    # Use regex to extract the desired numbers
    tar_file = re.search(tar_file_pattern, contents).group(1)
    total_images = re.search(total_images_pattern, contents).group(1)
    person_images = re.search(person_images_pattern, contents).group(1)
    # ratio = re.search(ratio_pattern, contents).group(1)
    # peak_memory = re.search(peak_memory_pattern, contents).group(1)
    time_cost = re.search(time_cost_pattern, contents).group(1)
    # avg_time = re.search(avg_time_pattern, contents).group(1)
    
    # Convert the extracted numbers to appropriate data types if needed
    tar_file_num += int(tar_file)
    total_images_num += int(total_images)
    person_images_num += int(person_images)
    # ratio = float(ratio)
    # peak_memory = float(peak_memory)
    time_cost_num += float(time_cost)
    # avg_time = float(avg_time)
    
    # Store the numbers in a list
    # numbers.append((total_images, person_images, ratio, peak_memory, time_cost, avg_time))
print(f"Total .tar file num is {tar_file_num}")
print(f"Total images num is {total_images_num}")
print(f"Total person images num is {person_images_num}")
print(f"Average human ratio is {person_images_num * 1. / total_images_num}")
print(f"Average time cost for each tar file is {time_cost_num / tar_file_num}")