#program to convert out dataset into the required format since flamingo uses webdataset 

import os
import json
import base64
import tarfile
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split

class DatasetConverter:
    def __init__(self, input_folder, output_folder, sequence_length=5, prediction_length=2, exclude_folders=None, chunk_size=10):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        if exclude_folders is None:
            exclude_folders = []
        self.exclude_folders = exclude_folders
        self.chunk_size = chunk_size
    
    def encode_image_to_base64(self, image_path):
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
    
    def process_sequences(self):
        all_sequences = []
        for subdir in os.listdir(self.input_folder):
            if subdir in self.exclude_folders:  # Skip excluded folders
                continue
                
            subdir_path = os.path.join(self.input_folder, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            descriptions_path = os.path.join(subdir_path, "image_descriptions.txt")
            if not os.path.exists(descriptions_path):
                continue

            descriptions = []
            with open(descriptions_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Skip blank lines
                        if "chunk" in line.lower():
                            descriptions += [None] * self.chunk_size
                        else:
                            descriptions.append(line)

            images = sorted([os.path.join(subdir_path, img) for img in os.listdir(subdir_path) if img.endswith((".jpg", ".png"))])
            # Ensure descriptions list is extended to match or exceed images count, handling excess images without descriptions
            descriptions += [None] * (len(images) - len(descriptions))

            for i in range(len(images) - self.sequence_length - self.prediction_length + 1):
                if None in descriptions[i:i + self.sequence_length + self.prediction_length]:
                    continue  # Skip sequences including or following 'chunk' placeholders
                
                sequence_images = images[i:i + self.sequence_length]
                future_annotations = descriptions[i + self.sequence_length:i + self.sequence_length + self.prediction_length]
                sequence_data = {
                    "images": [self.encode_image_to_base64(img) for img in sequence_images],
                    "annotations": future_annotations
                }
                all_sequences.append(sequence_data)
        print("sequences done")
        return all_sequences

    def split_data(self, sequences):
        # Split the dataset into train+val (80%) and test (20%)
        train_val, test = train_test_split(sequences, test_size=0.2, random_state=42)
        
        # Now split the 80% of train+val into train (80% of 80%) and val (20% of 80%)
        # This results in a 64% train, 16% val, and 20% test split of the original dataset
        train, val = train_test_split(train_val, test_size=0.2, random_state=42)
        
        return train, val, test
    
    def save_to_tar(self, sequences, filename):
        tar_path = os.path.join(self.output_folder, f"{filename}.tar")
        with tarfile.open(tar_path, "w") as tar:
            for i, sequence_data in enumerate(sequences):
                json_content = json.dumps(sequence_data)
                json_bytes = json_content.encode('utf-8')
                tarinfo = tarfile.TarInfo(name=f"{filename}_{i}.json")
                tarinfo.size = len(json_bytes)
                tar.addfile(tarinfo, BytesIO(json_bytes))
                print("tar added")

    def save_to_tar2(self, sequences, filename):
        chunk_size = self.chunk_size  # Number of sequences per .tar file
        num_chunks = len(sequences) // chunk_size + (1 if len(sequences) % chunk_size > 0 else 0)

        for chunk_index in range(num_chunks):
            start_index = chunk_index * chunk_size
            end_index = min(start_index + chunk_size, len(sequences))
            tar_path = os.path.join(self.output_folder, f"{filename}_{chunk_index:03d}.tar")

            with tarfile.open(tar_path, "w") as tar:
                for i, sequence_data in enumerate(sequences[start_index:end_index]):
                    json_content = json.dumps(sequence_data)
                    json_bytes = json_content.encode('utf-8')
                    tarinfo = tarfile.TarInfo(name=f"{filename}_{chunk_index:03d}_{i}.json")
                    tarinfo.size = len(json_bytes)
                    tar.addfile(tarinfo, BytesIO(json_bytes))
            print(f"Created {tar_path}")

    
    def convert(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        sequences = self.process_sequences()
        train_set, val_set, test_set = self.split_data(sequences)
        self.save_to_tar2(test_set, "test")
        self.save_to_tar2(train_set, "train")
        self.save_to_tar2(val_set, "validation")
        

# Example usage
input_folder = 'path to where data is present'
output_folder = 'expexted output folder'
exclude_folders = ['']
converter = DatasetConverter(input_folder, output_folder, sequence_length=5, prediction_length=2, exclude_folders=exclude_folders)
converter.convert()
