# -*- coding: utf-8 -*-
"""


設計のポイント:
- 「解析 (重い)」と「ランキング (軽い)」を分離。
  スライダーやジャンル以外の重み変更は再解析なしで即反映される。
- ジャンル除外はチェックボックスで指定 → 解析時に候補DBをフィルタ。
- 重みは4本のスライダーで自由に設定。合計が1でなくても内部で正規化。
- 🌟 メロディなし対応:
  動画のトーナリティ(音程感)が閾値未満の場合、クロマ(メロディ)重みを
  自動で減衰させ、削った分を他の指標へ比例再配分する。
  減衰の有無・理由はUI上に明示される。
"""

import os
import pickle
import subprocess
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import librosa
import imageio_ffmpeg
import tensorflow as tf
import tensorflow_hub as hub
import gradio as gr
from numpy.linalg import norm

warnings.filterwarnings('ignore')


# =========================================================
# 設定
# =========================================================
class Config:
    SR = 16000
    HOP_LENGTH = 512
    N_MFCC = 13
    TOP_N = 50
    TOP_K_KEYWORDS = 5

    MIN_WINDOW_SEC = 3
    WINDOW_GROWTH = 1.6
    MAX_WINDOW_CANDIDATES = 6
    LENGTH_BONUS = 0.03

    SILENCE_TOP_DB = 30
    MIN_ACTIVE_RATIO = 0.5
    IGNORE_KEYWORDS = ["Silence"]

    USE_CHROMA = True
    KEY_INVARIANT = True

    YAMNET_DB_FILE = 'fma_yamnet_db.pkl'
    MFCC_DB_FILE = 'music_mfcc_db.pkl'
    TRACKS_CSV_FILE = 'tracks.csv'
    KEYWORD_DB_FILE = 'music_yamnet_keywords_db.pkl'
    CHROMA_DB_FILE = 'music_chroma_db.pkl'


ALL_GENRES = ["Hip-Hop", "Pop", "Folk", "Experimental",
              "Rock", "International", "Electronic", "Instrumental"]

RESULT_TOP_K = 10  # ランキング表示件数


# =========================================================
# モデル・DBのロード (起動時に1回だけ)
# =========================================================
print("トラックのメタデータを読み込み中...")
tracks = pd.read_csv(Config.TRACKS_CSV_FILE, index_col=0, header=[0, 1])

print("YAMNetモデルをロード中...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
_cmp = yamnet_model.class_map_path().numpy()
_cmp = _cmp.decode('utf-8') if isinstance(_cmp, bytes) else str(_cmp)
class_names = pd.read_csv(_cmp)['display_name'].tolist()

print("データベースを読み込み中...")
with open(Config.YAMNET_DB_FILE, 'rb') as f:
    raw_yamnet_db = pickle.load(f)
with open(Config.MFCC_DB_FILE, 'rb') as f:
    raw_mfcc_db = pickle.load(f)

keyword_db = {}
if os.path.exists(Config.KEYWORD_DB_FILE):
    with open(Config.KEYWORD_DB_FILE, 'rb') as f:
        keyword_db = pickle.load(f)

chroma_db = {}
chroma_available = False
if Config.USE_CHROMA and os.path.exists(Config.CHROMA_DB_FILE):
    with open(Config.CHROMA_DB_FILE, 'rb') as f:
        chroma_db = pickle.load(f)
    chroma_available = True

# 曲名 -> ジャンル のキャッシュ (ジャンル除外を動的に切り替えるため)
def genre_of(song_name):
    try:
        track_id = int(song_name.split('.')[0])
        return tracks.loc[track_id, ('track', 'genre_top')]
    except Exception:
        return None

genre_cache = {name: genre_of(name) for name in raw_yamnet_db.keys()}
print(f" 準備完了 (全楽曲: {len(raw_yamnet_db)}曲 / クロマDB: {'あり' if chroma_available else 'なし'})")


# =========================================================
# 特徴抽出・スコア計算 (元コードのロジックを流用)
# =========================================================
def sec_to_frames(sec):
    return max(1, int((sec * Config.SR) / Config.HOP_LENGTH))


def frames_to_sec(frames):
    return float(librosa.frames_to_time(frames, sr=Config.SR, hop_length=Config.HOP_LENGTH))


def extract_target_features(filepath):
    temp_audio_path = "temp_search_audio.wav"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        cmd = [ffmpeg_exe, '-y', '-i', filepath, '-map', '0:1',
               '-ar', str(Config.SR), '-ac', '1', temp_audio_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            cmd = [ffmpeg_exe, '-y', '-i', filepath,
                   '-ar', str(Config.SR), '-ac', '1', temp_audio_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wav, _ = librosa.load(temp_audio_path, sr=Config.SR)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    if len(wav) == 0:
        return None

    wav_norm = wav / (np.max(np.abs(wav)) + 1e-8)
    scores, embeddings, _ = yamnet_model(wav_norm)
    yamnet_mean = np.mean(embeddings.numpy(), axis=0).astype(np.float32)

    predicted = tf.argmax(scores, axis=1).numpy()
    top_labels = [class_names[i] for i in predicted if class_names[i] not in Config.IGNORE_KEYWORDS]
    video_keywords = [w for w, _ in Counter(top_labels).most_common(Config.TOP_K_KEYWORDS)]

    mfcc = librosa.feature.mfcc(y=wav, sr=Config.SR, n_mfcc=Config.N_MFCC,
                                hop_length=Config.HOP_LENGTH).T
    mean, std = np.mean(mfcc, axis=0), np.std(mfcc, axis=0)
    mfcc_scaled = ((mfcc - mean) / (std + 1e-8)).astype(np.float32)

    rms = librosa.feature.rms(y=wav, hop_length=Config.HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    activity_mask = (rms_db > -Config.SILENCE_TOP_DB)
    n = mfcc_scaled.shape[0]
    if len(activity_mask) < n:
        activity_mask = np.pad(activity_mask, (0, n - len(activity_mask)), constant_values=False)
    activity_mask = activity_mask[:n]
    active_ratio = float(np.mean(activity_mask))

    video_chroma, tonality = None, 0.0
    if Config.USE_CHROMA:
        video_chroma = librosa.feature.chroma_cens(
            y=wav, sr=Config.SR, hop_length=Config.HOP_LENGTH).astype(np.float32)
        if video_chroma.shape[1] < n:
            video_chroma = np.pad(video_chroma, ((0, 0), (0, n - video_chroma.shape[1])))
        video_chroma = video_chroma[:, :n]
        mc = np.mean(video_chroma[:, activity_mask], axis=1) if np.any(activity_mask) \
            else np.mean(video_chroma, axis=1)
        p = mc / (np.sum(mc) + 1e-8)
        entropy = -np.sum(p * np.log2(p + 1e-8))
        tonality = float(1.0 - (entropy / np.log2(12)))

    return dict(yamnet=yamnet_mean, mfcc=mfcc_scaled, keywords=video_keywords,
                activity_mask=activity_mask, chroma=video_chroma,
                active_ratio=active_ratio, tonality=tonality)


def build_window_candidates(video_sec):
    candidates, w = [], float(Config.MIN_WINDOW_SEC)
    while len(candidates) < Config.MAX_WINDOW_CANDIDATES:
        wi = int(round(w))
        if wi > video_sec:
            break
        if not candidates or wi > candidates[-1]:
            candidates.append(wi)
        w *= Config.WINDOW_GROWTH
    return candidates or [max(1, int(video_sec))]


def make_chunks(target_mfcc_t, window_frames, activity_mask):
    target_len = target_mfcc_t.shape[1]
    chunks = []
    for i in range(0, target_len, window_frames):
        c = target_mfcc_t[:, i:i + window_frames]
        if c.shape[1] >= max(1, window_frames // 2) or not chunks:
            m = activity_mask[i:i + c.shape[1]]
            is_active = (np.mean(m) >= Config.MIN_ACTIVE_RATIO) if len(m) > 0 else False
            chunks.append((c, is_active, i))
    if chunks and not any(a for _, a, _ in chunks):
        chunks = [(c, True, i) for c, _, i in chunks]
    return chunks


def compute_sync_score(song_mfcc_t, target_chunks, step_frames):
    song_len = song_mfcc_t.shape[1]
    total = sum(c.shape[1] for c, _, _ in target_chunks)
    max_start = max(1, song_len - total + 1)
    best_sim, best_start = -1.0, 0

    for start_idx in range(0, max_start, step_frames):
        sims, cur = [], start_idx
        for t_chunk, is_active, _ in target_chunks:
            t_len = t_chunk.shape[1]
            if cur + t_len > song_len:
                break
            if is_active:
                s = song_mfcc_t[:, cur:cur + t_len]
                if s.shape[1] == t_len:
                    diff = t_chunk - s
                    dist = np.linalg.norm(diff) / np.sqrt(diff.size)
                    sims.append(np.exp(-dist * 1.5))
            cur += t_len
        if sims:
            boosted = (sum(sims) / len(sims)) * 0.6 + max(sims) * 0.4
            if boosted > best_sim:
                best_sim, best_start = boosted, start_idx
    return best_sim, best_start


def compute_chroma_sync_score(song_chroma, chunk_infos, step_frames):
    """クロマ(メロディ・和声)のシンクロ度を計算する。
    chunk_infos: [(mean_chroma_12, length, is_active, video_start_frame), ...]
    返り値: 最良シンクロ度 (0〜1, 失敗時は0)"""
    if song_chroma is None or not chunk_infos:
        return 0.0
    song_len = song_chroma.shape[1]
    total = sum(l for _, l, _, _ in chunk_infos)
    max_start = max(1, song_len - total + 1)
    csum = np.concatenate([np.zeros((12, 1)), np.cumsum(song_chroma, axis=1)], axis=1)
    best = 0.0
    for shift in range(12 if Config.KEY_INVARIANT else 1):
        shifted = []
        for mc, l, a, _ in chunk_infos:
            v = np.roll(mc, shift)
            shifted.append((v - np.mean(v), l, a))
        for start_idx in range(0, max_start, step_frames):
            sims, cur = [], start_idx
            for mc, l, a in shifted:
                if cur + l > song_len:
                    break
                if a:
                    seg = (csum[:, cur + l] - csum[:, cur]) / l
                    seg = seg - np.mean(seg)
                    denom = (np.linalg.norm(mc) * np.linalg.norm(seg)) + 1e-8
                    sims.append(max(float(np.dot(mc, seg) / denom), 0.0))
                cur += l
            if sims:
                boosted = (sum(sims) / len(sims)) * 0.6 + max(sims) * 0.4
                best = max(best, boosted)
    return best


def find_best_window_for_song(song_mfcc_t, chunks_by_window, step_frames):
    best = (-1.0, 0, None)
    best_raw = -1.0
    for win_sec, chunks in chunks_by_window.items():
        sim, start_idx = compute_sync_score(song_mfcc_t, chunks, step_frames)
        if sim < 0:
            continue
        bonus = 1.0 + Config.LENGTH_BONUS * np.log2(win_sec / Config.MIN_WINDOW_SEC) \
            if win_sec >= Config.MIN_WINDOW_SEC else 1.0
        if sim * bonus > best[0]:
            best = (sim * bonus, start_idx, win_sec)
            best_raw = sim
    return best_raw, best[1], best[2]


def normalize_metric(cands, key):
    vals = [c[key] for c in cands]
    vmin, vmax = min(vals), max(vals)
    rng = vmax - vmin
    for c in cands:
        c[key + '_norm'] = (c[key] - vmin) / rng if rng > 1e-8 else 0.5


# =========================================================
# 解析ステージ (重い処理。動画+ジャンル除外が変わったときだけ実行)
# =========================================================
def analyze_video(video_path, excluded_genres, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("動画ファイルをアップロードしてください。")

    # ジャンル除外を適用したDBを構築
    yamnet_db = {k: v for k, v in raw_yamnet_db.items()
                 if genre_cache.get(k) not in excluded_genres}
    if not yamnet_db:
        raise gr.Error("すべてのジャンルが除外されています。少なくとも1ジャンルは残してください。")
    mfcc_db = {k: raw_mfcc_db[k] for k in yamnet_db if k in raw_mfcc_db}

    progress(0.05, desc="音声を抽出して特徴量を計算中...")
    feat = extract_target_features(video_path)
    if feat is None:
        raise gr.Error("音声の抽出に失敗しました。音声トラックを含む動画か確認してください。")

    use_chroma = chroma_available and (feat['chroma'] is not None)

    # Stage 1: YAMNetで候補を絞り込み
    progress(0.25, desc="雰囲気(YAMNet)で候補を絞り込み中...")
    t_norm = norm(feat['yamnet'])
    stage1 = []
    for name, vec in yamnet_db.items():
        sim = np.dot(feat['yamnet'], vec) / (t_norm * norm(vec) + 1e-8)
        stage1.append((name, sim))
    stage1.sort(key=lambda x: x[1], reverse=True)
    top_candidates = stage1[:Config.TOP_N]

    # Stage 2: 窓長最適化つきシンクロ解析
    target_mfcc_t = feat['mfcc'].T
    video_sec = librosa.frames_to_time(target_mfcc_t.shape[1],
                                       sr=Config.SR, hop_length=Config.HOP_LENGTH)
    step_frames = sec_to_frames(1)
    window_candidates = build_window_candidates(video_sec)

    chunks_by_window, chroma_by_window = {}, {}
    for ws in window_candidates:
        chunks = make_chunks(target_mfcc_t, sec_to_frames(ws), feat['activity_mask'])
        chunks_by_window[ws] = chunks
        if use_chroma:
            infos = []
            for c, a, i in chunks:
                l = c.shape[1]
                infos.append((np.mean(feat['chroma'][:, i:i + l], axis=1), l, a, i))
            chroma_by_window[ws] = infos

    cands = []
    for idx, (name, y_sim) in enumerate(top_candidates):
        progress(0.3 + 0.65 * idx / len(top_candidates),
                 desc=f"シンクロ解析中... ({idx + 1}/{len(top_candidates)})")
        song_mfcc = mfcc_db[name]
        if isinstance(song_mfcc, dict):
            song_mfcc = song_mfcc.get('mfcc', song_mfcc)
        best_sim, best_start, best_win = find_best_window_for_song(
            song_mfcc.T, chunks_by_window, step_frames)

        # クロマ(メロディ)シンクロ度: MFCCで選ばれた最適窓の分割を使って評価
        chroma_score = 0.0
        if use_chroma and best_win in chroma_by_window:
            sc = chroma_db.get(name)
            if sc is not None:
                chroma_score = compute_chroma_sync_score(
                    sc, chroma_by_window[best_win], step_frames)

        # 🌟 キーワード一致の判定とカウント
        # 一致したキーワードを重複なしのリスト (matched_kws) として先に確定し、
        # 件数は必ず len(matched_kws) から導出する。
        # カウント用の変数を別々に持たない構成にすることで、
        # 「表示された一覧と件数が食い違う」余地をなくしている。
        song_kws = keyword_db.get(name, [])
        matched_kws = []
        raw = 0
        max_possible = sum((Config.TOP_K_KEYWORDS - i) ** 2 for i in range(Config.TOP_K_KEYWORDS))
        for vi, kw in enumerate(feat['keywords']):
            if kw in song_kws and kw not in matched_kws:
                si = song_kws.index(kw)
                raw += max(1, Config.TOP_K_KEYWORDS - vi) * max(1, Config.TOP_K_KEYWORDS - si)
                matched_kws.append(kw)
        matches = len(matched_kws)  # 🌟 件数は一覧の長さそのもの (常に一致する)
        kw_score = min(raw / max_possible, 1.0) if max_possible > 0 else 0.0

        cands.append({
            'name': name,
            'genre': genre_cache.get(name) or '-',
            'mfcc': max(best_sim, 0.0),
            'yamnet': float(y_sim),
            'keyword': kw_score,
            'chroma': chroma_score,
            'match_count': matches,
            'matched_keywords': matched_kws,          # 一致キーワード一覧 (重複なし)
            'time_sec': float(librosa.frames_to_time(best_start, sr=Config.SR,
                                                     hop_length=Config.HOP_LENGTH)),
            'window_sec': best_win if best_win is not None else '-',
        })

    for key in ('mfcc', 'yamnet', 'keyword', 'chroma'):
        normalize_metric(cands, key)

    state = {
        'candidates': cands,
        'use_chroma': use_chroma,
        'tonality': feat['tonality'],
        'active_ratio': feat['active_ratio'],
        'keywords': feat['keywords'],
        'video_sec': float(video_sec),
        'windows': window_candidates,
        'excluded': list(excluded_genres),
        'n_songs': len(yamnet_db),
    }

    info_md = (
        f"### 🎬 解析結果サマリー\n"
        f"- 動画の長さ: **{video_sec:.1f}秒** / 検索対象: **{len(yamnet_db)}曲** "
        f"(除外: {', '.join(excluded_genres) if excluded_genres else 'なし'})\n"
        f"- 有音フレーム率: **{feat['active_ratio'] * 100:.1f}%** (静寂部分はスコア計算から除外済み)\n"
        f"- 音程感 (トーナリティ): **{feat['tonality']:.2f}** (0=ノイズ的 / 1=明確な音程)\n"
        f"- 検出キーワード: {', '.join(feat['keywords']) if feat['keywords'] else '(なし)'}\n"
        f"- 試した窓長候補: {window_candidates} 秒\n"
        f"- クロマ(メロディ)軸: {' 有効' if use_chroma else ' 無効 (クロマDBなし)'}"
    )
    return state, info_md


# =========================================================
# ランキングステージ (軽い処理。重み変更のたびに即時実行)
# =========================================================
def effective_weights(w_mfcc, w_yamnet, w_kw, w_chroma, state,
                      auto_melody, tonality_threshold):
    """スライダー値 → 正規化 → メロディなし時の自動減衰・再配分 を行い、
    (実効重みdict, 説明メッセージ) を返す。"""
    w = {'mfcc': w_mfcc, 'yamnet': w_yamnet, 'keyword': w_kw, 'chroma': w_chroma}
    notes = []

    # クロマが物理的に使えない場合は強制0
    if not state['use_chroma'] and w['chroma'] > 0:
        notes.append("クロマDBが無いためメロディ重みは **0%** に固定されました。")
        w['chroma'] = 0.0

    # メロディなし判定: トーナリティが閾値未満なら比例減衰
    if state['use_chroma'] and auto_melody and w['chroma'] > 0:
        t = state['tonality']
        if t < tonality_threshold:
            scale = max(t, 0.0) / tonality_threshold  # 0〜1 の減衰率
            removed = w['chroma'] * (1.0 - scale)
            w['chroma'] *= scale
            notes.append(
                f"🎼 この動画は音程感が低い (トーナリティ {t:.2f} < 閾値 {tonality_threshold:.2f}) ため、"
                f"メロディ重みを **{scale * 100:.0f}%に減衰** し、残り{removed * 100:.0f}ptを他の指標へ再配分しました。"
            )
            # 削った分を他3指標へ比例再配分
            others = w['mfcc'] + w['yamnet'] + w['keyword']
            if others > 1e-8:
                for k in ('mfcc', 'yamnet', 'keyword'):
                    w[k] += removed * (w[k] / others)
            else:
                w['yamnet'] += removed  # 全て0なら雰囲気へ

    total = sum(w.values())
    if total < 1e-8:
        w = {'mfcc': 0.25, 'yamnet': 0.25, 'keyword': 0.25, 'chroma': 0.25 if state['use_chroma'] else 0.0}
        total = sum(w.values())
        notes.append("すべての重みが0だったため均等配分にしました。")
    w = {k: v / total for k, v in w.items()}
    return w, notes


def rank_candidates(state, w_mfcc, w_yamnet, w_kw, w_chroma,
                    auto_melody, tonality_threshold):
    if not state:
        return pd.DataFrame(), "まず動画を解析してください。"

    w, notes = effective_weights(w_mfcc, w_yamnet, w_kw, w_chroma,
                                 state, auto_melody, tonality_threshold)

    cands = state['candidates']
    for c in cands:
        c['final'] = (c['mfcc_norm'] * w['mfcc'] + c['yamnet_norm'] * w['yamnet'] +
                      c['keyword_norm'] * w['keyword'] + c['chroma_norm'] * w['chroma'])
    ranked = sorted(cands, key=lambda x: x['final'], reverse=True)[:RESULT_TOP_K]

    rows = []
    for i, c in enumerate(ranked):
        # 🌟 一致キーワードは「件数 + 一致した全キーワード」の形式で表示
        # 件数は matched_keywords の長さから直接計算するため、
        # 一覧に表示されるキーワード数と必ず一致する。
        # 例: "3件: Music, Speech, Guitar" / 一致なしなら "0件"
        mkws = c.get('matched_keywords', [])
        kw_disp = f"{len(mkws)}件: {', '.join(mkws)}" if mkws else "0件"
        rows.append({
            '順位': i + 1,
            '曲名': c['name'],
            'ジャンル': c['genre'],
            '総合スコア': round(c['final'], 4),
            'マッチ開始': f"{c['time_sec']:.1f}秒",
            '最適窓': f"{c['window_sec']}秒" if c['window_sec'] != '-' else '-',
            '一致キーワード': kw_disp,                       # 件数 + 一致した全KW
            '波形': round(c['mfcc_norm'], 2),
            '雰囲気': round(c['yamnet_norm'], 2),
            # 正規化値(候補内の相対値)だと「一致があるのに0.00」と表示され
            # 誤解を招くため、表示は生のキーワードスコア(絶対値)にする。
            # ランキング計算には従来どおり正規化値(keyword_norm)を使用。
            'キーワード': round(c['keyword'], 3),
            'メロディ': round(c['chroma_norm'], 2),
        })
    df = pd.DataFrame(rows)

    weight_md = (
        f"**実効重み** → 波形: {w['mfcc'] * 100:.0f}% | 雰囲気: {w['yamnet'] * 100:.0f}% | "
        f"キーワード: {w['keyword'] * 100:.0f}% | メロディ: {w['chroma'] * 100:.0f}%"
    )
    if notes:
        weight_md += "\n\n" + "\n\n".join(notes)
    return df, weight_md


# =========================================================
# プリセット
# =========================================================
PRESETS = {
    "⚖️ バランス型":        (0.15, 0.55, 0.15, 0.15),
    "🌊 展開重視":     (0.70, 0.10, 0.10, 0.10),
    "🎷 雰囲気重視": (0.10, 0.70, 0.10, 0.10),
    "📝 キーワード重視":  (0.20, 0.30, 0.50, 0.00),
    "🎼 メロディ重視": (0.10, 0.15, 0.05, 0.70),
}


def apply_preset(name):
    m, y, k, c = PRESETS[name]
    return m, y, k, c


# =========================================================
# UI 構築
# =========================================================
# プリセットのドロップダウンやジャンルのチェックボックスが
# 他の要素と違うフォントで表示されるのを防ぐため、
# UI全体に同一のフォントスタックを強制する
FONT_STACK = ("'Hiragino Kaku Gothic ProN', 'Hiragino Sans', 'Noto Sans JP', "
              "'Yu Gothic UI', 'Meiryo', system-ui, sans-serif")
CUSTOM_CSS = f"""
* , body, input, button, select, option, textarea,
label, .gr-check-radio, .gr-check-radio label span,
ul.options > li.item, .wrap .label-wrap span {{
    font-family: {FONT_STACK} !important;
}}
"""

with gr.Blocks(title="動画×楽曲シンクロ検索",
               theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Noto Sans JP"),
                                          "system-ui", "sans-serif"]),
               css=CUSTOM_CSS) as demo:
    gr.Markdown("# 動画×楽曲シンクロ検索\n"
                "動画をアップロードすると、雰囲気・波形・キーワード・メロディの4軸で"
                "シンクロする楽曲を検索します。重みはスライダーでリアルタイムに調整できます。")

    analysis_state = gr.State(None)

    with gr.Row():
        # ---------- 左カラム: 入力と解析 ----------
        with gr.Column(scale=1):
            video_in = gr.Video(label="解析する動画", sources=["upload"])
            genre_excl = gr.CheckboxGroup(
                choices=ALL_GENRES,
                value=["Experimental", "Electronic"],
                label="除外するジャンル",
                info="チェックしたジャンルの曲は検索対象から外れます",
            )
            analyze_btn = gr.Button("解析を実行", variant="primary")
            info_out = gr.Markdown("")

        # ---------- 右カラム: 重み調整とランキング ----------
        with gr.Column(scale=2):
            with gr.Accordion("重み調整 (合計は自動で100%に正規化されます)", open=True):
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()), value="⚖️ バランス型",
                    label="プリセット", info="選ぶとスライダーに反映されます")
                with gr.Row():
                    s_mfcc = gr.Slider(0, 1, value=0.15, step=0.05, label="波形 (MFCC)")
                    s_yamnet = gr.Slider(0, 1, value=0.55, step=0.05, label=" 雰囲気 (YAMNet)")
                with gr.Row():
                    s_kw = gr.Slider(0, 1, value=0.15, step=0.05, label="キーワード")
                    s_chroma = gr.Slider(0, 1, value=0.15, step=0.05, label="メロディ (クロマ)")
                with gr.Row():
                    auto_melody = gr.Checkbox(
                        value=True, label="メロディなし動画では自動的にメロディ重みを減衰する",
                        info="動画の音程感(トーナリティ)が閾値未満のとき、クロマ重みを減らして他へ再配分します")
                    tonality_th = gr.Slider(
                        0.0, 0.5, value=0.15, step=0.01, label="トーナリティ閾値",
                        info="この値未満なら「メロディなし」とみなす")

            weight_out = gr.Markdown("")
            result_df = gr.Dataframe(label=f"ランキング (上位{RESULT_TOP_K}曲)",
                                     interactive=False, wrap=True)

    # ---------- イベント配線 ----------
    rank_inputs = [analysis_state, s_mfcc, s_yamnet, s_kw, s_chroma, auto_melody, tonality_th]
    rank_outputs = [result_df, weight_out]

    analyze_btn.click(
        fn=analyze_video, inputs=[video_in, genre_excl],
        outputs=[analysis_state, info_out],
    ).then(fn=rank_candidates, inputs=rank_inputs, outputs=rank_outputs)

    preset_dd.change(fn=apply_preset, inputs=[preset_dd],
                     outputs=[s_mfcc, s_yamnet, s_kw, s_chroma])

    # スライダー・チェックボックスの変更は再解析なしで即ランキング更新
    for comp in (s_mfcc, s_yamnet, s_kw, s_chroma, auto_melody, tonality_th):
        comp.change(fn=rank_candidates, inputs=rank_inputs, outputs=rank_outputs)


if __name__ == "__main__":
    # inbrowser=True: 起動と同時に既定のブラウザで自動的にUIが開く
    # (URLを手入力・検索する必要がなくなる)
    demo.launch(inbrowser=True)
