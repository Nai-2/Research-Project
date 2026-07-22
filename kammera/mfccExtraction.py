
!pip install -q librosa tqdm scikit-learn
import os
import glob
import pickle
import warnings
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

# 警告非表示
warnings.filterwarnings("ignore")


class Config:
    SR = 16000
    HOP_LENGTH = 512
    N_MFCC = 13

    # 🎵 音楽ファイルがたくさん入っているDriveのフォルダパスを指定
    # 例: FMAデータセットなどを解凍したフォルダ
    MUSIC_DIR = '/content/drive/MyDrive/FMA_Dataset/fma_small'

    # 💾 保存先のファイル名
    OUTPUT_DB = '/content/drive/MyDrive/FMA_Dataset/music_mfcc_db.pkl'


def extract_features(file_path):
    """ 音声ファイルからMFCC系列と、その時間平均ベクトルを抽出する """
    # 音声の読み込み (モノラル, 16kHz)
    wav, _ = librosa.load(file_path, sr=Config.SR, mono=True)

    if len(wav) == 0:
        raise ValueError("無音または空のファイルです")

    # 1. MFCC（音色・展開の推移）を計算
    mfcc = librosa.feature.mfcc(y=wav, sr=Config.SR, n_mfcc=Config.N_MFCC, hop_length=Config.HOP_LENGTH)
    mfcc = mfcc.T  # (フレーム数, 13次元) に変換

    # 標準化 (各特徴量のスケールを揃える)
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc).astype(np.float32)

    # 2. 第1段階の爆速検索用: 動画全体の平均的な音色 (13次元ベクトル)
    # 時間軸(axis=0)方向に平均をとる
    mfcc_mean = np.mean(mfcc_scaled, axis=0).astype(np.float32)

    return mfcc_scaled, mfcc_mean


print(f"📂 フォルダ '{Config.MUSIC_DIR}' から音楽ファイルを検索中...")
# サブフォルダ内のmp3も再帰的に取得 (wavの場合は '*.wav' に変更してください)
music_files = glob.glob(f"{Config.MUSIC_DIR}/**/*.mp3", recursive=True)
print(f"✅ {len(music_files)} 曲のファイルを検出しました！")

music_db = {}
error_files = []

# ⚠️ テスト用に最初は100曲などで制限することをおすすめします
# LIMIT = 100
# target_files = music_files[:LIMIT]
target_files = music_files

print("\n🚀 特徴量(MFCC)の抽出とデータベース化を開始します...")
for path in tqdm(target_files):
    try:
        # 特徴量の抽出
        mfcc_seq, mfcc_mean = extract_features(path)

        # 辞書に格納
        # 構造: { "曲名.mp3": {"mfcc": [時間推移], "mean": [平均ベクトル]} }
        filename = os.path.basename(path)
        music_db[filename] = {
            "mfcc": mfcc_seq,
            "mean": mfcc_mean
        }
    except Exception as e:
        # 壊れているファイルや読み込めないファイルはスキップして記録
        error_files.append(path)

print(f"\n✅ 抽出完了: {len(music_db)} 曲のデータを辞書化しました。")
if error_files:
    print(f"⚠️ スキップされた破損ファイル数: {len(error_files)}")

# ==========================================
# 5. DBの保存 (Google Driveへ)
# ==========================================
print(f"💾 データベースをファイルに保存中... ({Config.OUTPUT_DB})")
with open(Config.OUTPUT_DB, 'wb') as f:
    pickle.dump(music_db, f)

print("✨ すべての作業が完了しました！")
print("この .pkl ファイルをPCにダウンロードして、検索スクリプトで使用してください。")
