import os
import pickle
import numpy as np
import librosa
import warnings
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import imageio_ffmpeg
import subprocess
from collections import Counter
from numpy.linalg import norm

warnings.filterwarnings('ignore')

class Config:
    SR = 16000
    HOP_LENGTH = 512
    N_MFCC = 13
    TOP_N = 30
    WINDOW_SEC = 20
    TOP_K_KEYWORDS = 5  # 🌟 キーワードの抽出個数（ここを変えればOK！）
    
    YAMNET_DB_FILE = 'fma_yamnet_db.pkl'
    MFCC_DB_FILE = 'music_mfcc_db.pkl'
    TRACKS_CSV_FILE = 'tracks.csv'
    KEYWORD_DB_FILE = 'music_yamnet_keywords_db.pkl'

EXCLUDE_GENRES = [] #<==除外するジャンル名を入れる

#fma_small に含まれるジャンルと曲数:
#Hip-Hop          1000
#Pop              1000
#Folk             1000
#Experimental     1000
#Rock             1000
#International    1000
#Electronic       1000
#Instrumental     1000

print("📂 トラックのメタデータを読み込み中...")
tracks = pd.read_csv(Config.TRACKS_CSV_FILE, index_col=0, header=[0, 1])

def is_music(song_name):
    try:
        track_id = int(song_name.split('.')[0])
        genre = tracks.loc[track_id, ('track', 'genre_top')]
        return genre not in EXCLUDE_GENRES
    except:
        return True 

print("🧠 YAMNetモデルをロード中...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path_raw = yamnet_model.class_map_path().numpy()
class_map_path = class_map_path_raw.decode('utf-8') if isinstance(class_map_path_raw, bytes) else str(class_map_path_raw)
class_names = pd.read_csv(class_map_path)['display_name'].tolist()

print("💾 データベースを読み込み中 (ジャンル除外適用中)...")
with open(Config.YAMNET_DB_FILE, 'rb') as f:
    raw_yamnet_db = pickle.load(f)
with open(Config.MFCC_DB_FILE, 'rb') as f:
    raw_mfcc_db = pickle.load(f)

keyword_db = {}
if os.path.exists(Config.KEYWORD_DB_FILE):
    with open(Config.KEYWORD_DB_FILE, 'rb') as f:
        keyword_db = pickle.load(f)
else:
    print(f"⚠️ {Config.KEYWORD_DB_FILE} が見つかりません。")

# メモリ上にフィルタリング済みのDBを保持
yamnet_db = {k: v for k, v in raw_yamnet_db.items() if is_music(k)}
mfcc_db = {k: v for k, v in raw_mfcc_db.items() if k in yamnet_db}

print(f"✅ すべての準備が完了しました！ (有効な楽曲: {len(yamnet_db)}曲)")

def extract_target_features(filepath):
    temp_audio_path = "temp_search_audio.wav"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    try:
        cmd = [ffmpeg_exe, '-y', '-i', filepath, '-map', '0:1', '-ar', str(Config.SR), '-ac', '1', temp_audio_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            cmd = [ffmpeg_exe, '-y', '-i', filepath, '-ar', str(Config.SR), '-ac', '1', temp_audio_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        wav, _ = librosa.load(temp_audio_path, sr=Config.SR)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    if len(wav) == 0: return None, None, []

    # YAMNet特徴量とキーワード
    wav_norm = wav / (np.max(np.abs(wav)) + 1e-8)
    scores, embeddings, _ = yamnet_model(wav_norm)
    yamnet_mean = np.mean(embeddings.numpy(), axis=0).astype(np.float32)
    
    predicted_classes = tf.argmax(scores, axis=1).numpy()
    top_labels = [class_names[idx] for idx in predicted_classes]
    video_keywords = [word for word, freq in Counter(top_labels).most_common(Config.TOP_K_KEYWORDS)]
    
    # MFCC特徴量
    mfcc = librosa.feature.mfcc(y=wav, sr=Config.SR, n_mfcc=Config.N_MFCC, hop_length=Config.HOP_LENGTH).T
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)
    mfcc_scaled = ((mfcc - mean) / (std + 1e-8)).astype(np.float32)
    
    return yamnet_mean, mfcc_scaled, video_keywords


def execute_multi_profile_search(target_video_file):
    print(f"\n🎬 動画 [{target_video_file}] の解析を開始します...")
    
    target_yamnet, target_mfcc, video_keywords = extract_target_features(target_video_file)
    if target_yamnet is None:
        print("❌ 音声抽出に失敗しました。")
        return
    
    print(f"🏷️ 動画のキーワード: {video_keywords}")
    target_norm = norm(target_yamnet)

    # [Stage 1] YAMNetによる全体絞り込み (ここは共通)
    stage1_scores = []
    for song_name, song_yamnet in yamnet_db.items():
        song_norm = norm(song_yamnet)
        sim = np.dot(target_yamnet, song_yamnet) / (target_norm * song_norm + 1e-8)
        stage1_scores.append((song_name, sim))
    stage1_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = stage1_scores[:Config.TOP_N]

    # [Stage 2] 局所特徴とキーワードの算出 (重み付け前に全候補の各スコアを計算)
    target_mfcc_t = target_mfcc.T
    compare_len = min(target_mfcc_t.shape[1], int((Config.WINDOW_SEC * Config.SR) / Config.HOP_LENGTH))
    target_sub = target_mfcc_t[:, :compare_len]
    step_frames = int(1 * Config.SR / Config.HOP_LENGTH)
    
    candidate_metrics = []
    
    for song_name, yamnet_sim in top_candidates:
        # MFCCの波形シンクロスコア算出
        song_mfcc = mfcc_db[song_name].get('mfcc', mfcc_db[song_name]) if isinstance(mfcc_db[song_name], dict) else mfcc_db[song_name]
        song_mfcc_t = song_mfcc.T
        
        best_local_sim, best_start_idx = -1.0, 0
        for start_idx in range(0, max(1, song_mfcc_t.shape[1] - compare_len), step_frames):
            song_sub = song_mfcc_t[:, start_idx : start_idx + compare_len]
            if song_sub.shape[1] == compare_len:
                diff = target_sub - song_sub
                dist = np.linalg.norm(diff) / compare_len
                local_sim = np.exp(-dist * 1.5)
                if local_sim > best_local_sim:
                    best_local_sim = local_sim
                    best_start_idx = start_idx
        
        # キーワード一致スコア算出 (上位から重み付け)
        song_keywords = keyword_db.get(song_name, [])
        keyword_score_raw = 0
        match_count = 0
        
        # 完全一致の場合の最大理論スコアを計算 (例: 10位までなら 100+81+64...=385)
        max_possible_score = sum((Config.TOP_K_KEYWORDS - idx) ** 2 for idx in range(Config.TOP_K_KEYWORDS))
        
        for v_idx, v_kw in enumerate(video_keywords):
            if v_kw in song_keywords:
                s_idx = song_keywords.index(v_kw)
                # 動画と楽曲それぞれの順位（1位=10点, 2位=9点...）を掛け合わせて加算
                weight_v = max(1, Config.TOP_K_KEYWORDS - v_idx)
                weight_s = max(1, Config.TOP_K_KEYWORDS - s_idx)
                keyword_score_raw += (weight_v * weight_s)
                match_count += 1
                
        # 獲得スコアを最大スコアで割って 0.0 〜 1.0 に正規化
        keyword_score = min(keyword_score_raw / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        
        best_time_sec = librosa.frames_to_time(best_start_idx, sr=Config.SR, hop_length=Config.HOP_LENGTH)
        
        # 3つの生の指標を保存
        candidate_metrics.append({
            'name': song_name,
            'mfcc': best_local_sim,
            'yamnet': yamnet_sim,
            'keyword': keyword_score,
            'match_count': match_count,
            'time_sec': best_time_sec
        })

    # 様々な重みのプロファイル（合計1.0になるように設定）
    profiles = [
        {"name": "⚖️ バランス型", "w_mfcc": 0.15, "w_yamnet": 0.7, "w_kw": 0.15},
        {"name": "🌊 波形・展開重視", "w_mfcc": 0.8, "w_yamnet": 0.1, "w_kw": 0.1},
        {"name": "🎷 雰囲気(YAMNet)重視", "w_mfcc": 0.1, "w_yamnet": 0.8, "w_kw": 0.1},
        {"name": "📝 キーワード意味重視", "w_mfcc": 0.2, "w_yamnet": 0.3, "w_kw": 0.5}
    ]

    for profile in profiles:
        print(f"\n【{profile['name']}】 (波形:{profile['w_mfcc']*100}% | 雰囲気:{profile['w_yamnet']*100}% | キーワード:{profile['w_kw']*100}%)")
        print(f"{'順位':<3} | {'総合スコア':<8} | {'曲名':<20} | {'マッチ開始秒'} | {'KW一致'}")
        print("-" * 65)
        
        # このプロファイルの重みでスコアを計算
        for cand in candidate_metrics:
            cand['final_score'] = (cand['mfcc'] * profile['w_mfcc']) + \
                                  (cand['yamnet'] * profile['w_yamnet']) + \
                                  (cand['keyword'] * profile['w_kw'])
                                  
        # スコア順にソートして上位3曲を表示
        candidate_metrics.sort(key=lambda x: x['final_score'], reverse=True)
        
        for i, cand in enumerate(candidate_metrics[:3]):
            print(f"{i+1:<4} | {cand['final_score']:.4f}     | {cand['name']:<20} | {cand['time_sec']:5.1f}秒付近 | {cand['match_count']}件")

execute_multi_profile_search('804042734.923077.mp4')
