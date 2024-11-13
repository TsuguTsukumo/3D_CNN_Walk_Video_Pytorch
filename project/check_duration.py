import os
from moviepy.editor import VideoFileClip

# 動画が入っているフォルダのパスを指定
folder_path = '/workspace/data/test_side_and_front/fold0/val/ASD_not'

# フォルダ内のファイルを取得
files = os.listdir(folder_path)

total_duration = 0  # 合計時間を記録する変数
video_count = 0     # 動画ファイル数をカウントする変数

# 動画ファイルの長さを取得して表示
for file_name in files:
    if file_name.endswith('.mp4'):  # mp4ファイルのみを対象
        video_path = os.path.join(folder_path, file_name)
        clip = VideoFileClip(video_path)
        duration = clip.duration  # 動画の長さ（秒）
        print(f"{file_name}: {duration:.2f} seconds")
        total_duration += duration  # 合計時間に加算
        video_count += 1  # 動画ファイル数をカウント

# 合計時間とファイル数を表示
print(f"\nTotal duration: {total_duration:.2f} seconds")
print(f"Total number of video files: {video_count}")
