import os
import shutil
import json
import random

# 入力データセットの構造
root_dir = "/workspace/data/data/Combined_video"

# JSONファイルのパス
split_json_file_path = "/workspace/data/data/Combined_video/split_results.json"

# 出力ディレクトリ
output_root_dir = "/workspace/data/data/Combined_video_by_json_4th"

# JSONファイルを読み込み
with open(split_json_file_path, "r") as f:
    splits = json.load(f)

# 元のデータセットのパスからファイルをコピーする関数
def copy_files_for_split(split_data, split_type, fold_number):
    for person_data in split_data:
        category_name = person_data["category"]
        person_id = person_data["person_id"]

        # 元のデータセットのパス
        original_path = os.path.join(root_dir, category_name, person_id)

        # 出力ディレクトリのパス（foldとtrainまたはvalidフォルダ）
        output_fold_dir = os.path.join(output_root_dir, f"fold_{fold_number}")
        output_category_dir = os.path.join(output_fold_dir, split_type, category_name)
        os.makedirs(output_category_dir, exist_ok=True)
        
        output_person_dir = os.path.join(output_category_dir, person_id)
        os.makedirs(output_person_dir, exist_ok=True)

        # 元のフォルダ内の内容を新しいフォルダにコピー
        if os.path.exists(original_path):
            for item in os.listdir(original_path):
                original_item_path = os.path.join(original_path, item)
                output_item_path = os.path.join(output_person_dir, item)
                if os.path.isdir(original_item_path):
                    shutil.copytree(original_item_path, output_item_path)
                else:
                    shutil.copy2(original_item_path, output_item_path)
        else:
            print(f"警告: ディレクトリが見つかりません - {original_path}")

# 各foldに対してファイルをコピー
def create_folds():
    for split_name, split_data in splits.items():
        print(f"処理中: {split_name}")

        # Split_X (Xはfold番号) の場合を想定
        fold_number = int(split_name.split('_')[-1])  # Split_1 -> 1
        
        # トレーニングセットと検証セットのフォルダを作成
        for split_type in ["train", "val"]:
            # 対応するデータを取得（train または valid）
            split_type_data = split_data[split_type + "_data"]
            
            # JSONファイルの情報に基づいてファイルをコピー
            copy_files_for_split(split_type_data, split_type, fold_number)

    print("フォルダとデータのコピーが完了しました！")

# 5-fold交差検証用のフォルダ作成
create_folds()