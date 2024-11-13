import cv2
import os

# 動画ファイルが保存されているフォルダのパス
input_folder = 'workspace/data/test_side_and_front/fold0/val/ASD_not'
# 分割した動画を保存するフォルダのパス
output_folder = '/workspace/data/Test_1_only_side_error/fold0/val/ASD_not'

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# 指定したフォルダ内の全ての動画ファイルを処理
for filename in os.listdir(input_folder):
    if filename.endswith('.mp4'):  # mp4ファイルを対象
        input_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)  # フレームレートを取得
        frame_interval = int(fps * 2)  # 2秒ごとのフレーム数

        index = 1  # 保存する動画ファイルのインデックス
        frames = []  # 分割したフレームを保存するリスト

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 動画の最後に達した場合はループを終了

            frames.append(frame)  # フレームをリストに追加

            # 2秒ごとにフレームを保存
            if len(frames) == frame_interval:
                output_filename = f"{os.path.splitext(filename)[0]}_{index}.mp4"
                output_path = os.path.join(output_folder, output_filename)

                # 新しい動画ファイルを作成
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

                for f in frames:
                    out.write(f)  # フレームを書き込む

                out.release()  # 出力ファイルを閉じる
                frames = []  # フレームリストをリセット
                index += 1  # インデックスをインクリメント

        cap.release()  # 入力ファイルを閉じる

print("動画の分割が完了しました。")