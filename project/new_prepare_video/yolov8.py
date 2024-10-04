import cv2
from ultralytics import YOLO
import os
import torch

# Ensure CUDA device is specified
torch.cuda.set_device(1)
# モデルをロード
model = YOLO(model='yolov8n.pt').to('cuda')

# ディレクトリの設定
root_dir = '/workspace/data/origin/non-ASD/HipOA'
output_dir = '/workspace/data/resized/HipOA'

# 信頼度スコアの閾値
confidence_threshold = 0.5

# 出力先ディレクトリを作成
os.makedirs(output_dir, exist_ok=True)

# 指定されたディレクトリで再帰的にfull_ap.mp4を探して処理
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower() == 'full_ap.mp4':
            video_path = os.path.join(subdir, file)

            # 出力動画のファイル名を「サブディレクトリ名 + full_ap.mp4」に設定
            directory_name = os.path.basename(subdir)  # サブディレクトリ名を取得
            output_video_path = os.path.join(output_dir, f'{directory_name}_full_ap.mp4')  # 保存ファイル名を作成

            # 既に同じファイルが存在する場合は処理をスキップ
            if os.path.exists(output_video_path):
                print(f"Skipping {output_video_path} (already exists)")
                continue

            cap = cv2.VideoCapture(video_path)

            # 出力動画の設定
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック設定
            fps = 30  # フレームレート設定
            frame_size = (512, 512)  # リサイズ後のフレームサイズ
            out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 画像をモデルに渡して推論（人のみを検出対象にし、信頼度閾値を設定）
                results = model(frame, classes=[0], conf=confidence_threshold)  # クラスID 0（人）のみを検出

                # 検出結果を取得
                detections = results[0].boxes

                # 人のバウンディングボックスを保存するリスト
                person_boxes = []

                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]

                    # 信頼度が閾値以上のバウンディングボックスのみ使用
                    if conf >= confidence_threshold:
                        person_boxes.append([x1, y1, x2, y2])

                # 人が検出された場合のみ処理
                if person_boxes:
                    # 左側に位置する人のバウンディングボックスを切り出す
                    leftmost_box = min(person_boxes, key=lambda b: b[0])
                    x1, y1, x2, y2 = map(int, leftmost_box)
                    cropped_person = frame[y1:y2, x1:x2]

                    # 切り出した画像を512x512にリサイズ
                    resized_person = cv2.resize(cropped_person, frame_size)

                    # リサイズした画像を動画に追加
                    out.write(resized_person)

            # リソースの解放
            cap.release()
            out.release()
            print(f"Processed: {output_video_path}")
