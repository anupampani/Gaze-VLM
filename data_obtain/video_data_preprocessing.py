#author-anupam 
#obtain frames and corresponding descriptions 

import cv2
import os 

def get_frames(data_path ,save_path):
  os.makedirs(save_path, exist_ok=True)
  for ann_file in os.listdir(os.path.join(data_path, 'gaze')):
    if ann_file == 'manifest.csv' or ann_file == 'manifest.ver':
        continue
    vid = ann_file.split('.')[0]
    vidcap = cv2.VideoCapture(os.path.join(data_path, 'full_scale', f'{vid}.mp4'))
      # Check if video capture was successful
    if not vidcap.isOpened():
        print(f"Error: Could not open video file {vid}. Skipping...")
        continue  # Skip to the next file
    
    new_save_path= os.path.join(save_path,f'{vid}')
    os.makedirs(new_save_path,exist_ok=True)

    success,image = vidcap.read()
    seconds=0;
    count = 0
    while success:
      if (count % 30 ==0):
        frame_filename = os.path.join(new_save_path, f'{seconds}.jpg')
        cv2.imwrite(frame_filename,image)     # save frame as JPEG file 
        seconds= seconds +1     
      success,image = vidcap.read()
      count += 1
    print('Video to frame for one video complete')

def main():
    path_to_ego4d = ''  # change this to your own path

    data_path = path_to_ego4d
    save_path = f'{path_to_ego4d}/output folder'
    get_frames(data_path=data_path, save_path=save_path)
 


if __name__ == '__main__':
    main()