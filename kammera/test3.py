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
    TOP_N = 30  # Yamnetで絞り込む候補曲数
    TOP_K_KEYWORDS = 5  # キーワードの抽出個数

    # 🌟 窓長の自動最適化に関する設定
    MIN_WINDOW_SEC = 3       # 最短の窓長（これ未満にはしない）
    WINDOW_GROWTH = 1.6      # 候補窓長を増やす倍率 (3 -> 4.8 -> 7.7 -> ...)
    MAX_WINDOW_CANDIDATES = 6  # 試す窓長の最大個数
    LENGTH_BONUS = 0.03      # 窓長が2倍になるごとに加えるスコアボーナス率
                             # (0.0 にすると純粋なシンクロ度だけで窓長を選ぶ)

    YAMNET_DB_FILE = 'fma_yamnet_db.pkl'
    MFCC_DB_FILE = 'music_mfcc_db.pkl'
    TRACKS_CSV_FILE = 'tracks.csv'
    KEYWORD_DB_FILE = 'music_yamnet_keywords_db.pkl'

EXCLUDE_GENRES = ["Experimental", "Electronic"]  # <==除外するジャンル名を入れる

# fma_small に含まれるジャンルと曲数:
# Hip-Hop          1000
# Pop              1000
# Folk             1000
# Experimental     1000
# Rock             1000
# International    1000
# Electronic       1000
# Instrumental     1000

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


def sec_to_frames(sec):
    return max(1, int((sec * Config.SR) / Config.HOP_LENGTH))


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

    if len(wav) == 0:
        return None, None, []

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


def build_window_candidates(video_sec):
    """
    🌟 動画の長さに応じて、試す窓長（秒）の候補リストを自動生成する。
    最短は MIN_WINDOW_SEC (=3秒)。そこから WINDOW_GROWTH 倍ずつ増やし、
    動画全長を超えない範囲で最大 MAX_WINDOW_CANDIDATES 個まで作る。
    例: 動画25秒 -> [3, 5, 8, 12, 20] のような候補になる。
    """
    candidates = []
    w = float(Config.MIN_WINDOW_SEC)
    while len(candidates) < Config.MAX_WINDOW_CANDIDATES:
        w_int = int(round(w))
        if w_int > video_sec:
            break
        if not candidates or w_int > candidates[-1]:
            candidates.append(w_int)
        w *= Config.WINDOW_GROWTH

    # 動画が3秒未満の場合など、候補が空なら動画全体を1つの窓として使う
    if not candidates:
        candidates = [max(1, int(video_sec))]
    return candidates


def make_chunks(target_mfcc_t, window_frames):
    """動画のMFCC系列を window_frames ごとのチャンクに分割する"""
    target_len_frames = target_mfcc_t.shape[1]
    chunks = []
    for i in range(0, target_len_frames, window_frames):
        c = target_mfcc_t[:, i:i + window_frames]
        # 末尾の極端に短い端数チャンク（窓の半分未満）はノイズになりやすいので捨てる
        if c.shape[1] >= max(1, window_frames // 2) or not chunks:
            chunks.append(c)
    return chunks


def compute_sync_score(song_mfcc_t, target_chunks, step_frames):
    """
    分割済みチャンク列を楽曲上でスライドさせ、最良のシンクロ度と開始位置を返す。
    距離は「要素数の平方根で正規化したRMS距離」を使うことで、
    窓長が違ってもスコアを公平に比較できるようにしている。
    (元コードの norm/フレーム数 だと長い窓ほど距離が小さく出て不公平になるため)
    """
    song_len_frames = song_mfcc_t.shape[1]
    total_target_frames = sum(c.shape[1] for c in target_chunks)
    max_start = max(1, song_len_frames - total_target_frames + 1)

    best_sim, best_start = -1.0, 0

    for start_idx in range(0, max_start, step_frames):
        chunk_sims = []
        curr_start = start_idx

        for t_chunk in target_chunks:
            t_len = t_chunk.shape[1]
            if curr_start + t_len > song_len_frames:
                break
            s_chunk = song_mfcc_t[:, curr_start:curr_start + t_len]
            if s_chunk.shape[1] == t_len:
                diff = t_chunk - s_chunk
                # 🌟 長さ非依存のRMS距離 (窓長を変えても比較可能)
                dist = np.linalg.norm(diff) / np.sqrt(diff.size)
                sim = np.exp(-dist * 1.5)
                chunk_sims.append(sim)
            curr_start += t_len

        if chunk_sims:
            avg_sim = sum(chunk_sims) / len(chunk_sims)
            max_sim = max(chunk_sims)
            # 平均 60% + 最大値 40%: 「一部が神がかって合う」曲を評価しやすくする
            boosted_sim = (avg_sim * 0.6) + (max_sim * 0.4)
            if boosted_sim > best_sim:
                best_sim = boosted_sim
                best_start = start_idx

    return best_sim, best_start


def find_best_window_for_song(song_mfcc_t, chunks_by_window, step_frames):
    """
    🌟 曲ごとの窓長最適化の本体。
    すべての候補窓長についてシンクロ度を計算し、
    「長さボーナス込みのスコア」が最大になる窓長を採用する。
    返り値: (シンクロ度, 開始フレーム, 採用した窓長[秒])
    """
    best = (-1.0, 0, None)  # (adjusted_sim, start_idx, window_sec)
    best_raw_sim = -1.0

    for win_sec, chunks in chunks_by_window.items():
        sim, start_idx = compute_sync_score(song_mfcc_t, chunks, step_frames)
        if sim < 0:
            continue

        # 窓長ボーナス: スコアが拮抗しているときは「長く似ている」方を優先
        bonus = 1.0 + Config.LENGTH_BONUS * np.log2(win_sec / Config.MIN_WINDOW_SEC) \
            if win_sec >= Config.MIN_WINDOW_SEC else 1.0
        adjusted = sim * bonus

        if adjusted > best[0]:
            best = (adjusted, start_idx, win_sec)
            best_raw_sim = sim

    # 最終スコアとしてはボーナス抜きの生シンクロ度を返す（表示・合成用）
    return best_raw_sim, best[1], best[2]


def normalize_metric_across_candidates(candidates, key):
    """
    🌟 指標を候補曲の中で 0.0〜1.0 に min-max 正規化する。
    生スコアのスケール（distの定義など）に関わらず、各指標が
    フルレンジで順位に寄与するようにし、プロファイルの重みを意図通り効かせる。
    全候補が同値の場合は差がない指標なので一律 0.5 とする（順位に影響しない）。
    """
    vals = [c[key] for c in candidates]
    vmin, vmax = min(vals), max(vals)
    rng = vmax - vmin
    for c in candidates:
        c[key + '_norm'] = (c[key] - vmin) / rng if rng > 1e-8 else 0.5


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

    # [Stage 2] 曲ごとに窓長を最適化しながら波形シンクロ解析 ＆ キーワード評価
    target_mfcc_t = target_mfcc.T
    target_len_frames = target_mfcc_t.shape[1]
    video_sec = librosa.frames_to_time(target_len_frames, sr=Config.SR, hop_length=Config.HOP_LENGTH)
    step_frames = sec_to_frames(1)

    # 🌟 窓長の候補を生成し、各候補でのチャンク分割を先に1回だけ済ませておく
    window_candidates = build_window_candidates(video_sec)
    chunks_by_window = {}
    for win_sec in window_candidates:
        window_frames = sec_to_frames(win_sec)
        chunks_by_window[win_sec] = make_chunks(target_mfcc_t, window_frames)

    print(f"🔍 [Stage 2] 曲ごとに最適な窓長を探索してシンクロ解析中...")
    print(f"   (試す窓長の候補: {window_candidates} 秒 / 動画の長さ: {video_sec:.1f}秒)")

    candidate_metrics = []

    for song_name, yamnet_sim in top_candidates:
        song_mfcc = mfcc_db[song_name].get('mfcc', mfcc_db[song_name]) if isinstance(mfcc_db[song_name], dict) else mfcc_db[song_name]
        song_mfcc_t = song_mfcc.T

        # 🌟 この曲にとって最もシンクロ度が高くなる窓長を自動選択
        best_local_sim, best_start_idx, best_window_sec = find_best_window_for_song(
            song_mfcc_t, chunks_by_window, step_frames
        )

        # キーワード一致スコア算出 (上位から重み付け)
        song_keywords = keyword_db.get(song_name, [])
        keyword_score_raw = 0
        match_count = 0

        max_possible_score = sum((Config.TOP_K_KEYWORDS - idx) ** 2 for idx in range(Config.TOP_K_KEYWORDS))

        for v_idx, v_kw in enumerate(video_keywords):
            if v_kw in song_keywords:
                s_idx = song_keywords.index(v_kw)
                weight_v = max(1, Config.TOP_K_KEYWORDS - v_idx)
                weight_s = max(1, Config.TOP_K_KEYWORDS - s_idx)
                keyword_score_raw += (weight_v * weight_s)
                match_count += 1

        keyword_score = min(keyword_score_raw / max_possible_score, 1.0) if max_possible_score > 0 else 0.0

        best_time_sec = librosa.frames_to_time(best_start_idx, sr=Config.SR, hop_length=Config.HOP_LENGTH)

        candidate_metrics.append({
            'name': song_name,
            'mfcc': max(best_local_sim, 0.0),
            'yamnet': yamnet_sim,
            'keyword': keyword_score,
            'match_count': match_count,
            'time_sec': best_time_sec,
            'window_sec': best_window_sec if best_window_sec is not None else '-'
        })

    # 🌟 3指標を候補間で 0〜1 に正規化 (これをしないとスケールの狭い指標が順位に効かなくなる)
    normalize_metric_across_candidates(candidate_metrics, 'mfcc')
    normalize_metric_across_candidates(candidate_metrics, 'yamnet')
    normalize_metric_across_candidates(candidate_metrics, 'keyword')

    # 様々な重みのプロファイル（合計1.0になるように設定）
    profiles = [
        {"name": "⚖️ バランス型", "w_mfcc": 0.15, "w_yamnet": 0.7, "w_kw": 0.15},
        {"name": "🌊 波形・展開重視", "w_mfcc": 0.8, "w_yamnet": 0.1, "w_kw": 0.1},
        {"name": "🎷 雰囲気(YAMNet)重視", "w_mfcc": 0.1, "w_yamnet": 0.8, "w_kw": 0.1},
        {"name": "📝 キーワード意味重視", "w_mfcc": 0.2, "w_yamnet": 0.3, "w_kw": 0.5}
    ]

    for profile in profiles:
        print(f"\n【{profile['name']}】 (波形:{profile['w_mfcc']*100}% | 雰囲気:{profile['w_yamnet']*100}% | キーワード:{profile['w_kw']*100}%)")
        print(f"{'順位':<3} | {'総合スコア':<8} | {'曲名':<20} | {'マッチ開始秒'} | {'最適窓':<5} | {'KW一致'}")
        print("-" * 78)

        for cand in candidate_metrics:
            # 🌟 正規化済みの指標 (*_norm) を使うことで、各プロファイルの重みが意図通りに効く
            cand['final_score'] = (cand['mfcc_norm'] * profile['w_mfcc']) + \
                                  (cand['yamnet_norm'] * profile['w_yamnet']) + \
                                  (cand['keyword_norm'] * profile['w_kw'])

        candidate_metrics.sort(key=lambda x: x['final_score'], reverse=True)

        for i, cand in enumerate(candidate_metrics[:3]):
            win_disp = f"{cand['window_sec']}秒" if cand['window_sec'] != '-' else '-'
            print(f"{i+1:<4} | {cand['final_score']:.4f}     | {cand['name']:<20} | {cand['time_sec']:5.1f}秒付近 | {win_disp:<6} | {cand['match_count']}件")

#execute_multi_profile_search('IMG_6974.MOV')  # ここに解析したい動画ファイルのパスを指定
if __name__ == "__main__":
    execute_multi_profile_search("IMG_7081.MOV")