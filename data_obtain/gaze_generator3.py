import sys
sys.path.append('/home/anupam/projects/RAFT/core')

from collections import defaultdict
import argparse
import torch
import torch.nn.functional as F
import os
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use('Agg')

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate
from tqdm import tqdm

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def calculate_optical_flow(model, img1, img2):
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
    return flow_low, flow_up

def upsample_flow_component(flow_component, factor=8):
    flow_component = torch.tensor(flow_component).float()
    flow_component = flow_component.unsqueeze(0).unsqueeze(0)
    up_flow_component = F.interpolate(flow_component, scale_factor=factor, mode='bilinear', align_corners=True)
    up_flow_component = up_flow_component.squeeze(0).squeeze(0)
    up_flow_component = up_flow_component * factor
    return up_flow_component.numpy()

def adjust_coordinates(norm_pos_x, norm_pos_y, img_width=1088, img_height=1080):
    x = norm_pos_x * img_width
    y = (1 - norm_pos_y) * img_height
    return x, y

def adjust_gaze_with_flow(gaze_points, flow_x, flow_y):
    adjusted_points = []
    for x, y in gaze_points:
        new_x = x + flow_x[int(round(y)), int(round(x))]
        new_y = y + flow_y[int(round(y)), int(round(x))]
        adjusted_points.append((new_x, new_y))
    return adjusted_points

def generate_heatmap(points, img_width, img_height, sigma=50):
    heatmap = np.zeros((img_height, img_width))
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < img_width and 0 <= y < img_height:
            heatmap[y, x] += 1
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap

def save_heatmap_image(heatmap, output_path):
    heatmap_normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
    img = Image.fromarray(heatmap_normalized)
    img.save(output_path)

def process_optical_flow_and_gaze_correction(raft_model, frames_folder, gaze_file_path, output_folder, img_width=1088, img_height=1080, fps=30, delta_t=0.2):
    # frames = sorted(os.listdir(frames_folder))
    # Existing code
    frames = sorted(os.listdir(frames_folder), key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

    frames_by_second = defaultdict(list)
    print(frames)

    # Group frames by second
    for frame in frames:
        second, frame_number = map(int, frame.split('_')[0]), int(frame.split('_')[1].split('.')[0])
        frames_by_second[second].append(frame)
    threshold = 10
    max_diff_proportion = 0.4

    gaze_data = {}
    with open(gaze_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row['frame'])
            x, y = adjust_coordinates(float(row['x']), float(row['y']), img_width, img_height)
            gaze_data[frame] = (x, y)

    all_corrected_gaze_points = []

    for second in sorted(frames_by_second.keys()):  # Ensure seconds are processed in order
        frames = sorted(frames_by_second[second], key=lambda x: int(x.split('_')[1].split('.')[0]))
        last_frame_path = os.path.join(frames_folder, frames[-1])
        last_frame = load_image(last_frame_path)
        print(last_frame_path)

        for frame in frames[:-1]:  # Exclude the last frame
            frame_path = os.path.join(frames_folder, frame)
            current_frame = load_image(frame_path)

            torch.cuda.empty_cache()

            flow_low1, flow_up1 = calculate_optical_flow(raft_model, current_frame, last_frame)
            flow_low2, flow_up2 = calculate_optical_flow(raft_model, last_frame, current_frame)
            flow_x1, flow_y1, flow1 = forward_interpolate(flow_low1[0])
            flow_x2, flow_y2, flow2 = forward_interpolate(flow_low2[0])

            diff_x = torch.abs(flow_x1 + flow_x2)
            diff_y = torch.abs(flow_y1 + flow_y2)

            num_diff_x = torch.sum(diff_x > threshold).item()
            num_diff_y = torch.sum(diff_y > threshold).item()
            total_pixels = flow_x1.numel()

            proportion_diff_x = num_diff_x / total_pixels
            proportion_diff_y = num_diff_y / total_pixels

            if proportion_diff_x > max_diff_proportion or proportion_diff_y > max_diff_proportion:
                continue

            flow_x1 = upsample_flow_component(flow_x1)
            print("checkpoint2")
            flow_y1 = upsample_flow_component(flow_y1)
            print("checkpoint3")

            frame_num = second * fps + int(frame.split('_')[1].split('.')[0])
            if frame_num in gaze_data:
                corrected_gaze = adjust_gaze_with_flow([gaze_data[frame_num]], flow_x1, flow_y1)[0]
                all_corrected_gaze_points.append(corrected_gaze)

    for second in range(0, len(frames) // fps):
        start_frame = max(0, (second + 1) * fps - int(delta_t * fps))
        end_frame = (second + 1) * fps
        corrected_points_for_second = [pt for idx, pt in enumerate(all_corrected_gaze_points) if start_frame <= idx < end_frame]
        if corrected_points_for_second:
            heatmap = generate_heatmap(corrected_points_for_second, img_width, img_height, sigma=20)
            save_heatmap_image(heatmap, os.path.join(output_folder, f'{second + 1}.jpg'))

def process_all_videos(root_folder, output_root_folder, gaze_folder, args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    print("starting process")

    for video_folder in os.listdir(root_folder):
        frames_folder = os.path.join(root_folder, video_folder)
        gaze_file_path = os.path.join(gaze_folder, f'{video_folder}_frame_label.csv')
        output_folder = os.path.join(output_root_folder, video_folder)

        if os.path.isdir(frames_folder) and os.path.isfile(gaze_file_path):
            os.makedirs(output_folder, exist_ok=True)
            process_optical_flow_and_gaze_correction(model, frames_folder, gaze_file_path, output_folder)
            
        torch.cuda.empty_cache()  # Clear GPU memory after processing each video

if __name__ == '__main__':
    root_folder = 'folder with indivial images'
    gaze_folder = 'folder with gaze csv files'
    output_root_folder = 'required output folder'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()

    process_all_videos(root_folder, output_root_folder, gaze_folder, args)
