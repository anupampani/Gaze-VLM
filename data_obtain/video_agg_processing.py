#17 july 
#Author: anupam - code for breaking down video into image frames depending on delta size

import os
import cv2

def get_frames(data_path, save_path, delta_t):
    os.makedirs(save_path, exist_ok=True)
    for ann_file in os.listdir(os.path.join(data_path, 'full_scale.gaze')):
        if ann_file == 'manifest.csv' or ann_file == 'manifest.ver':
            continue
        vid = ann_file.split('.')[0]
        vidcap = cv2.VideoCapture(os.path.join(data_path, 'full_scale.gaze', f'{vid}.mp4'))
        
        # Check if video capture was successful
        if not vidcap.isOpened():
            print(f"Error: Could not open video file {vid}. Skipping...")
            continue  # Skip to the next file
        
        new_save_path = os.path.join(save_path, f'{vid}')
        os.makedirs(new_save_path, exist_ok=True)

        success, image = vidcap.read()
        count = 0
        fps = 30  # Assuming the video has 30 frames per second
        frame_range = int(delta_t * fps)  # Calculate the range of frames to capture

        while success:
            second = count // fps
            start_frame = max(0, (second + 1) * fps - frame_range)
            end_frame = (second + 1) * fps

            if start_frame <= count < end_frame:
                frame_in_second = count - start_frame
                frame_filename = os.path.join(new_save_path, f'{second + 1}_{frame_in_second + 1}.jpg')
                cv2.imwrite(frame_filename, image)  # Save frame as JPEG file
                print(f'Saved frame: {frame_filename}')
            
            success, image = vidcap.read()
            count += 1

        print(f'Video to frame for {vid} complete')

def main():
    path_to_ego4d = '/mnt/lv1/ego4d/v2/Ego4D'  # Change this to your own path
    delta_t = 0.4  # Change this to your desired delta_t value

    data_path = path_to_ego4d
    save_path = f'{path_to_ego4d}/save path'
    get_frames(data_path=data_path, save_path=save_path, delta_t=delta_t)

if __name__ == '__main__':
    main()
