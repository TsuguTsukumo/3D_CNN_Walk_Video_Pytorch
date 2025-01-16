import os
import shutil

# 入力データセットのルートディレクトリ
root_dir = "/workspace/data/data/Combined_video_by_json/train"

# 出力先ディレクトリ
out_dir = "/workspace/data/data/Cross_Validation/ex_20250116/train"

# ルートディレクトリ内のカテゴリごとに処理
for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    
    # 出力先カテゴリディレクトリを作成
    category_out_path = os.path.join(out_dir, category)
    os.makedirs(category_out_path, exist_ok=True)
    
    # 各日付のディレクトリを処理
    for date in os.listdir(category_path):
        date_path = os.path.join(category_path, date)
        
        # 日付ごとの動画ファイルをリネームしてコピー
        if os.path.isdir(date_path):
            video_files = [f for f in os.listdir(date_path) if f.endswith('.mp4')]
            
            for i, video_file in enumerate(video_files):
                # 新しいファイル名を作成
                new_video_name = f"{date}_{i+1}.mp4"
                new_video_path = os.path.join(category_out_path, new_video_name)
                
                # ファイルを新しい場所にコピー
                shutil.copy(os.path.join(date_path, video_file), new_video_path)

print("データセットの再構成（コピー）が完了しました！")
