import os
import glob
import pickle
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from collections import Counter
import concurrent.futures
from tqdm.notebook import tqdm
from google.colab import drive

# ==========================================
# 0. Google Driveのマウント
# ==========================================
print("📁 Google Driveをマウントします...")
drive.mount('/content/drive')

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
# 🌟 ご自身の環境に合わせてフォルダのパスを変更してください
AUDIO_DIR = '/content/drive/MyDrive/FMA_Dataset/fma_small' # FMAの音楽ファイルが入っているフォルダ
OUTPUT_PKL = '/content/drive/MyDrive/FMA_Dataset/music_yamnet_keywords_db.pkl' # データベースの保存先

SR = 16000 # YAMNetの要求するサンプリングレート
TOP_K_KEYWORDS = 10 # 各楽曲から抽出するキーワードの最大数

# ==========================================
# 2. YAMNetモデルのロードとクラス名の準備
# ==========================================
print("🧠 YAMNetモデルをロード中...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')

def load_class_names(class_map_csv):
    df = pd.read_csv(class_map_csv)
    return df['display_name'].tolist()

class_names = load_class_names(class_map_path)

# ==========================================
# 3. GPUメモリの最適化 (フル読み込み時のOOM対策)
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPUメモリを必要な分だけ動的に確保する設定（メモリ溢れを防ぐ）
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 GPUが有効になっています！高速並列処理を開始します。")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPUが見つかりません。CPUで処理します。（遅くなる可能性があります）")

# ==========================================
# 4. キーワード抽出処理関数
# ==========================================
def process_single_file(filepath):
    filename = os.path.basename(filepath)
    try:
        # 🌟 音楽ファイルをフルコーラスで読み込み (時間指定なし)
        wav, _ = librosa.load(filepath, sr=SR)
        if len(wav) == 0:
            return filename, None

        # 正規化してYAMNetに入力 (全編の音声がここで一気にGPUで計算されます)
        wav_norm = wav / (np.max(np.abs(wav)) + 1e-8)
        scores, embeddings, spectrogram = yamnet_model(wav_norm)

        # 各フレームで最も確率の高いクラス(音の種類)を取得
        predicted_classes = tf.argmax(scores, axis=1).numpy()
        top_labels = [class_names[idx] for idx in predicted_classes]

        # 曲全体を通して出現頻度の高いトップ10をキーワードとして採用
        counts = Counter(top_labels)
        song_keywords = [word for word, freq in counts.most_common(TOP_K_KEYWORDS)]

        return filename, song_keywords
    except Exception as e:
        # メモリ不足など致命的なエラーの時だけ表示
        print(f"Error processing {filename}: {e}")
        return filename, None

# ==========================================
# 5. 全体処理の実行 (並列処理)
# ==========================================
def process_audio_files():
    audio_files = glob.glob(os.path.join(AUDIO_DIR, '**/*.mp3'), recursive=True)
    print(f"🎵 見つかった楽曲ファイル: {len(audio_files)}件")

    if len(audio_files) == 0:
        print("❌ 楽曲ファイルが見つかりません。AUDIO_DIR のパス設定を確認してください。")
        return

    keyword_db = {}

    # フルコーラスを並列でGPUに投げるとメモリが溢れる(OOM)可能性があるため、
    # スレッド数は 2 程度に抑えるのが無料枠のGPU環境(T4 16GB)では一番安定して速いです。
    max_workers = 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 全ファイルをタスクとして登録
        futures = {executor.submit(process_single_file, filepath): filepath for filepath in audio_files}

        # tqdmでプログレスバー(進捗)を表示しながら、完了したものから結果を受け取る
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(audio_files), desc="並列キーワード抽出中"):
            filename, keywords = future.result()
            if keywords is not None:
                keyword_db[filename] = keywords

    # データベースの保存
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(keyword_db, f)

    print(f"\n✅ キーワード抽出完了！ {len(keyword_db)}件のデータを保存しました。")
    print(f"📁 保存先: {OUTPUT_PKL}")

# 実行
if __name__ == "__main__":
    process_audio_files()
