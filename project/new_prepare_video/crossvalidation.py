import os
import random
import shutil
from moviepy.editor import VideoFileClip
from sklearn.model_selection import KFold

def get_video_metadata(folder_path):
    videos = []
    for label in ['ASD', 'ASD_not']:
        label_path = os.path.join(folder_path, label)
        for file in os.listdir(label_path):
            if file.endswith('.mp4'):
                file_path = os.path.join(label_path, file)
                try:
                    with VideoFileClip(file_path) as video:
                        duration = video.duration
                        videos.append({
                            "file_path": file_path,
                            "duration": duration,
                            "label": label
                        })
                except Exception as e:
                    print(f"Could not process {file_path}: {e}")
    return videos

def save_split_videos(videos, output_folder, n_splits=5):
    random.shuffle(videos)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(videos)):
        train_videos = [videos[i] for i in train_idx]
        val_videos = [videos[i] for i in val_idx]
        
        fold_output_path = os.path.join(output_folder, f'fold{fold}')
        train_output_path = os.path.join(fold_output_path, 'train')
        val_output_path = os.path.join(fold_output_path, 'val')
        
        for path in [train_output_path, val_output_path]:
            os.makedirs(os.path.join(path, 'ASD'), exist_ok=True)
            os.makedirs(os.path.join(path, 'ASD_not'), exist_ok=True)
        
        save_videos(train_videos, train_output_path, "Train")
        save_videos(val_videos, val_output_path, "Validation")

def save_videos(video_list, destination, set_type):
    total_duration = 0
    for video in video_list:
        total_duration += video['duration']
        dest_folder = os.path.join(destination, video['label'])
        shutil.copy(video['file_path'], dest_folder)
    print(f"{set_type} set: {len(video_list)} videos, Total duration: {total_duration:.2f} seconds")

# メイン処理
input_folder_path = "/workspace/data/data/new_data_selected_cropped copy"  # 入力ディレクトリ
output_folder_path = "/workspace/data/data/new_data_cropped_cross_validation"  # 出力ディレクトリ

videos = get_video_metadata(input_folder_path)
save_split_videos(videos, output_folder_path, n_splits=5)
