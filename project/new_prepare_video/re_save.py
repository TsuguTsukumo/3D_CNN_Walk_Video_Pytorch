import os
import shutil

def save_and_rename_videos(input_dir, output_dir):
    # 入力ディレクトリ内を再帰的に探索
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                # 現在のファイルのパス
                current_file_path = os.path.join(root, file)
                
                # ディレクトリ構造から病名と日時を取得
                relative_path = os.path.relpath(root, input_dir)
                parts = relative_path.split(os.sep)
                
                if len(parts) >= 2:
                    disease_name = parts[-2]
                    date = parts[-1]
                else:
                    print(f"病名または日時が見つかりません: {current_file_path}")
                    continue
                
                # 保存先のファイル名を生成
                new_file_name = f"{disease_name}_{date}_{file}"
                save_dir = os.path.join(output_dir, disease_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # 新しい保存先のパス
                new_file_path = os.path.join(save_dir, new_file_name)
                
                # ファイルをコピーしてリネーム
                shutil.copy2(current_file_path, new_file_path)
                print(f"Saved: {new_file_path}")

# 入力ディレクトリと出力ディレクトリを指定
input_dir = '/workspace/data/data/new_data_selected_cropped'  # 病名/日時/full_ap.mp4 などが格納されている
output_dir = '/workspace/data/data/new_data_selected_cropped copy/ASD_not'  # リネーム後のファイルを保存する場所

save_and_rename_videos(input_dir, output_dir)
