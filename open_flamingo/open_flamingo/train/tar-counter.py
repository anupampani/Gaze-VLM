import glob
import os

# Set the path to your dataset directory
dataset_dir = 'insert path/datasets/dataset_gaze_caption'

# Pattern to match all .tar files for the training dataset
train_pattern = os.path.join(dataset_dir, 'test*.tar')

# Use glob to find all files matching the pattern
train_files = glob.glob(train_pattern)

# Sort the list of train .tar files
train_files_sorted = sorted(train_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

# Print the count of train .tar files
print(f"Train .tar files count: {len(train_files_sorted)}")

# Print the first 10 train .tar files' names, in ascending order
# print("First 10 train .tar files (in ascending order):")
# for train_file in train_files_sorted[:]:
#     print(os.path.basename(train_file))




#new dataset - less obs
#train - 6789 (0,6788)


#train - 0 to 6759
#test - 0 to 2112
#validation - 0 to 1689
#size is 55G of the new dataset
#52 for the old 

# train : 6758 (0 to 6757)
# test : 2112 (0 to 2111 )
# validation: 1690 (0 to 1689)
