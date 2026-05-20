#できてるかわかんないが推薦システム本体（Gemini作）
#特徴量ファイルはなんかアップロードできなかったんで各自作ってくれると助かる

import time
import queue
import pickle
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import warnings
from numpy.linalg import norm

# 警告非表示
warnings.filterwarnings('ignore')

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
class Config:
    SR = 16000                 
    DURATION = 20               
    CHUNK_SIZE = 16000         
    
    # Colabから持ってきた辞書ファイルのみ指定
    DB_FILE = 'fma_yamnet_db.pkl'

# ==========================================
# 2. 特徴量データベース マネージャー
# ==========================================
class VectorMatchManager:
    def __init__(self):
        print("💾 特徴量データベース(pkl)を読み込み中...")
        try:
            with open(Config.DB_FILE, 'rb') as f:
                self.music_db = pickle.load(f)
            print(f"✅ ロード完了: {len(self.music_db)} 曲のベクトルデータをメモリに展開しました")
        except FileNotFoundError:
            print(f"❌ エラー: {Config.DB_FILE} が見つかりません。同じフォルダに配置してください。")
            exit()

    def find_best_match(self, env_vector):
        best_match = None
        highest_sim = -1.0
        
        # Numpyを使った高速なコサイン類似度総当たり計算
        env_norm = norm(env_vector)
        if env_norm == 0: return None, 0.0

        for song_name, song_vector in self.music_db.items():
            song_norm = norm(song_vector)
            if song_norm == 0: continue
            
            sim = np.dot(env_vector, song_vector) / (env_norm * song_norm)
            
            if sim > highest_sim:
                highest_sim = sim
                best_match = song_name
                
        return best_match, highest_sim

# ==========================================
# 3. YAMNetマネージャー (実機推論用)
# ==========================================
class YAMNetManager:
    def __init__(self):
        print("🧠 YAMNetモデルを読み込み中 (マイク推論用)...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("✅ モデル準備完了")

    def extract_vector(self, waveform):
        """ マイク音声から1024次元の環境音ベクトルを抽出 """
        _, embeddings, _ = self.model(waveform)
        # 時間軸で平均化して1次元のベクトルに圧縮
        env_vector = np.mean(embeddings.numpy(), axis=0)
        return env_vector

# ==========================================
# 4. マイクストリーミング処理
# ==========================================
class AudioStreamer:
    def __init__(self):
        self.q = queue.Queue()
        self.buffer = np.zeros(Config.SR * Config.DURATION, dtype=np.float32)
        
    def callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy().flatten())

    def update_buffer(self):
        while not self.q.empty():
            new_data = self.q.get()
            self.buffer = np.roll(self.buffer, -len(new_data))
            self.buffer[-len(new_data):] = new_data

# ==========================================
# 5. メインループ
# ==========================================
def main():
    matcher = VectorMatchManager()
    yamnet = YAMNetManager()
    streamer = AudioStreamer()
    
    stream = sd.InputStream(
        samplerate=Config.SR, channels=1, dtype=np.float32, 
        blocksize=Config.CHUNK_SIZE, callback=streamer.callback
    )
    
    print("\n🎤 マイク監視開始 (軽量テキスト推論モード)")
    print("-------------------------------------------------")
    
    current_playing = None
    
    with stream:
        try:
            while True:
                time.sleep(8)  # 8秒ごとに環境音をサンプリング
                streamer.update_buffer()
                
                print("🔄 潜在空間で波形を検索中...")
                # 1. 環境音を1024次元ベクトルに変換
                env_vector = yamnet.extract_vector(streamer.buffer)
                
                # 2. 辞書と突き合わせて最も波形が近い曲を検索
                best_match, similarity = matcher.find_best_match(env_vector)
                
                if best_match:
                    if best_match != current_playing:
                        print(f"🎧 [マッチ成功] 👉 推奨トラック: 『 {best_match} 』 (類似度: {similarity:.4f})")
                        current_playing = best_match
                    else:
                        print(f"    (類似度: {similarity:.4f} / ムード継続: {best_match})")
                else:
                    print("🎧 (マッチする楽曲が見つかりませんでした)")
                    
                print("-------------------------------------------------")
                
        except KeyboardInterrupt:
            print("\n⏹️ システムを停止しました。")

if __name__ == "__main__":
    main()
