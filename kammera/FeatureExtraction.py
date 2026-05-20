#zipをダウンロードする空きがなかったのでcolabで特徴量データだけ作ったやつ（Gemini作）


# ==========================================
# 1. 必要なライブラリのインストールとインポート
# ==========================================
!pip install -q librosa tqdm

import os
import glob
import pickle
import warnings
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm.notebook import tqdm
from google.colab import files

# 壊れたMP3ファイル読み込み時の警告を非表示
warnings.filterwarnings("ignore")

# ==========================================
# 2. fma_small データセットのダウンロードと展開
# ==========================================
# ※ fma_small.zip は約7.2GBあります。ダウンロードと解凍に数分かかります。
if not os.path.exists('fma_small'):
    print("🌍 FMA Smallデータセット(7.2GB)をダウンロード中...")
    !curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
    print("📦 zipファイルを展開中...")
    !unzip -q fma_small.zip
    print("✅ データセットの準備完了")
else:
    print("✅ データセットは既に存在します")

# ==========================================
# 3. YAMNetモデルのロード
# ==========================================
print("🧠 YAMNetモデルをロード中...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# ==========================================
# 4. 全楽曲の波形解析とベクトル化
# ==========================================
# 解凍されたフォルダからMP3ファイルのパスをすべて取得
all_mp3_files = glob.glob('fma_small/*/*.mp3')
print(f"🎵 合計 {len(all_mp3_files)} 曲のMP3ファイルが見つかりました")

# ⚠️ テスト用の制限（まずは100曲でテスト作成をお勧めします）
# フルデータ(8000曲)で回す場合は None にしてください
LIMIT_TRACKS = None
if LIMIT_TRACKS:
    mp3_files_to_process = all_mp3_files[:LIMIT_TRACKS]
    print(f"⏱️ テストモード: 先頭の {LIMIT_TRACKS} 曲のみを処理します")
else:
    mp3_files_to_process = all_mp3_files

music_features_db = {}
error_files = []

print("\n🚀 特徴量ベクトル（1024次元）の抽出を開始します...")
for file_path in tqdm(mp3_files_to_process):
    try:
        # YAMNetの仕様(16kHz, モノラル)に合わせて読み込み
        # FMAには一部壊れたMP3が混ざっているためtry-exceptで保護
        wav_data, _ = librosa.load(file_path, sr=16000, mono=True)

        # モデルに入力して推論 (scores, embeddings, log_mel_spectrogram)
        _, embeddings, _ = yamnet_model(wav_data)

        # 時間軸で平均化し、曲全体を1つの1024次元ベクトルに圧縮
        song_vector = np.mean(embeddings.numpy(), axis=0)

        # ファイル名（例: 000002.mp3）をキーにして保存
        song_name = os.path.basename(file_path)
        music_features_db[song_name] = song_vector

    except Exception as e:
        error_files.append(file_path)

print(f"\n✅ 抽出完了: {len(music_features_db)} 曲のベクトル化に成功しました")
if error_files:
    print(f"⚠️ 読み込みエラー(壊れたMP3等): {len(error_files)} 曲")

# ==========================================
# 5. DBファイルの保存とダウンロード
# ==========================================
DB_FILENAME = 'fma_yamnet_db.pkl'

with open(DB_FILENAME, 'wb') as f:
    pickle.dump(music_features_db, f)

print(f"💾 データベースを {DB_FILENAME} に保存しました。PCにダウンロードします...")
files.download(DB_FILENAME)
