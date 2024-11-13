import os
import shutil
from moviepy.editor import VideoFileClip
from sklearn.model_selection import StratifiedKFold
import numpy as np

# ディレクトリのパスを指定
asd_dir = '/workspace/data/data_for_poster/raw_data'
not_asd_dir = '/workspace/data/data_for_poster/raw_data/ASD_not'
output_dir = 'workspace/data/data_for_poster_retry'

# full_ap.mp4 のパスと動画の長さを取得する関数
def get_video_paths_and_durations(data_dir, label, disease_name=None):
    video_paths = []
    durations = []
    
    # ディレクトリが直接ラベルのものか病名のサブフォルダを持つかを確認
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        
        if os.path.isdir(folder_path):  # 病名のサブフォルダ (LCS or DHS)
            for date_folder in os.listdir(folder_path):
                full_ap_path = os.path.join(folder_path, date_folder, 'full_ap.mp4')
                if os.path.exists(full_ap_path):
                    video_paths.append((full_ap_path, date_folder, label, folder))  # 病名を含める
                    clip = VideoFileClip(full_ap_path)
                    durations.append(clip.duration)
                    clip.close()
        else:  # 通常のASDデータ
            full_ap_path = os.path.join(data_dir, folder, 'full_ap.mp4')
            if os.path.exists(full_ap_path):
                video_paths.append((full_ap_path, folder, label))
                clip = VideoFileClip(full_ap_path)
                durations.append(clip.duration)
                clip.close()

    return video_paths, durations

# ASDと非ASDの動画パスと長さを取得
asd_video_paths, asd_durations = get_video_paths_and_durations(asd_dir, 'ASD')
not_asd_video_paths, not_asd_durations = get_video_paths_and_durations(not_asd_dir, 'ASD_not')

# 動画パス、時間、ラベルを統合
video_paths = asd_video_paths + not_asd_video_paths
durations = np.array(asd_durations + not_asd_durations)
labels = np.array([1] * len(asd_video_paths) + [0] * len(not_asd_video_paths))

# 動画時間を考慮して4:1に分割する関数
# 動画時間を考慮して4:1に分割する関数
# 動画時間を考慮して4:1に分割する関数
# 動画時間を考慮して4:1に分割する関数
def time_and_count_based_split(video_paths, durations, labels, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(video_paths, labels)):
        train_paths = [video_paths[i] for i in train_idx]
        val_paths = [video_paths[i] for i in val_idx]
        train_durations = durations[train_idx]
        val_durations = durations[val_idx]

        # 時間の比率を調整する
        target_ratio = 4 / (4 + 1)  # 4:1の目標比率
        train_total_duration = np.sum(train_durations)
        val_total_duration = np.sum(val_durations)

        # 初期の比率計算
        current_ratio = val_total_duration / (train_total_duration + val_total_duration)
        
        # 動画時間比率が4:1に近づくように調整
        while current_ratio > (1 - target_ratio) and len(val_paths) > 0:
            # 検証セットからトレーニングセットへ動画を移動
            max_val_idx = np.argmax(val_durations)  # 最も長い動画を選択
            train_paths.append(val_paths[max_val_idx])  # トレーニングに移動
            train_durations = np.append(train_durations, val_durations[max_val_idx])

            # バリデーションセットから削除
            val_paths.pop(max_val_idx)
            val_durations = np.delete(val_durations, max_val_idx)

            # 時間を再計算
            train_total_duration = np.sum(train_durations)
            val_total_duration = np.sum(val_durations)
            current_ratio = val_total_duration / (train_total_duration + val_total_duration)

        # 各フォールドの時間比率を表示
        print(f"Fold {fold_num + 1}:")
        print(f"  Train total duration: {train_total_duration:.2f} seconds")
        print(f"  Val total duration: {val_total_duration:.2f} seconds")
        print(f"  Ratio (Val/Total): {current_ratio:.2f}")

        splits.append((train_paths, val_paths))
    
    return splits



# データをコピーする関数
def save_split_data(split, fold_num):
    train_paths, val_paths = split

    fold_dir = os.path.join(output_dir, f'fold{fold_num}')
    train_dir = os.path.join(fold_dir, 'train')
    val_dir = os.path.join(fold_dir, 'val')

    # トレーニングとバリデーションのディレクトリを作成
    for category in ['ASD', 'ASD_not']:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    # トレーニングデータをコピー
    for video_path, date_folder, label, *disease_name in train_paths:
        if disease_name:  # 非ASDデータの場合は病名を追加
            disease_label = disease_name[0]
            dest_folder = os.path.join(train_dir, label)
            dest_path = os.path.join(dest_folder, f"{date_folder}_full_ap_{disease_label}.mp4")
        else:  # ASDデータの場合
            dest_folder = os.path.join(train_dir, label)
            dest_path = os.path.join(dest_folder, f"{date_folder}_full_ap.mp4")
        shutil.copy(video_path, dest_path)

    # バリデーションデータをコピー
    for video_path, date_folder, label, *disease_name in val_paths:
        if disease_name:  # 非ASDデータの場合は病名を追加
            disease_label = disease_name[0]
            dest_folder = os.path.join(val_dir, label)
            dest_path = os.path.join(dest_folder, f"{date_folder}_full_ap_{disease_label}.mp4")
        else:  # ASDデータの場合
            dest_folder = os.path.join(val_dir, label)
            dest_path = os.path.join(dest_folder, f"{date_folder}_full_ap.mp4")
        shutil.copy(video_path, dest_path)

# 動画時間と人数を考慮してデータを5分割
splits = time_and_count_based_split(video_paths, durations, labels, n_splits=5)

# 各フォールドのデータを保存
for fold_num, split in enumerate(splits):
    save_split_data(split, fold_num)
    print(f"Fold {fold_num + 1} completed.")

print("クロスバリデーション用のデータ分割と保存が完了しました。")
