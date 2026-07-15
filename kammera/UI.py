import tkinter as tk
from tkinter import scrolledtext, font
import threading
import time
import sounddevice as sd

# 元のシステムファイルをインポート
import test as mr

class RecommenderUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 環境音 音楽推薦システム")
        self.root.geometry("600x500")
        self.root.configure(bg="#f4f4f9")

        self.is_running = False
        self.thread = None

        # フォント設定
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.song_font = font.Font(family="Helvetica", size=18, weight="bold")
        
        self.setup_ui()
        
        # システムの初期化（少し重いので画面描画後に実行）
        self.root.after(100, self.initialize_system)

    def setup_ui(self):
        # --- ステータス＆操作パネル ---
        top_frame = tk.Frame(self.root, bg="#f4f4f9", pady=10)
        top_frame.pack(fill=tk.X)

        self.status_label = tk.Label(top_frame, text="ステータス: 準備中...", font=self.title_font, bg="#f4f4f9", fg="#333")
        self.status_label.pack(side=tk.LEFT, padx=20)

        self.start_btn = tk.Button(top_frame, text="▶ スタート", font=("Helvetica", 12), bg="#4caf50", fg="white", state=tk.DISABLED, command=self.start_monitoring)
        self.start_btn.pack(side=tk.RIGHT, padx=10)

        self.stop_btn = tk.Button(top_frame, text="⏹ 停止", font=("Helvetica", 12), bg="#f44336", fg="white", state=tk.DISABLED, command=self.stop_monitoring)
        self.stop_btn.pack(side=tk.RIGHT, padx=10)

        # --- 現在の推奨トラック表示 ---
        mid_frame = tk.Frame(self.root, bg="#ffffff", bd=2, relief=tk.GROOVE)
        mid_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(mid_frame, text="現在のおすすめトラック", font=("Helvetica", 10), bg="#ffffff", fg="#666").pack(pady=(10, 0))
        
        self.current_song_label = tk.Label(mid_frame, text="---", font=self.song_font, bg="#ffffff", fg="#1976d2")
        self.current_song_label.pack(pady=(5, 15))

        # --- ログ出力エリア ---
        log_frame = tk.Frame(self.root, bg="#f4f4f9")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        tk.Label(log_frame, text="システムログ", bg="#f4f4f9").pack(anchor=tk.W)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled', bg="#2b2b2b", fg="#a9b7c6", font=("Consolas", 10))
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def initialize_system(self):
        self.log_message("システムを初期化中... (モデルとDBのロード)")
        self.root.update()
        try:
            # 元コードのクラスをインスタンス化
            # 音声特徴ベクトルと楽曲データベースを突き合わせるマネージャー
            self.matcher = mr.VectorMatchManager()
            # YAMNetを使って音を特徴ベクトルに変換
            self.yamnet = mr.YAMNetManager()
            #マイクから入ってくる音声を一時的に保存するバッファを管理するクラス
            self.streamer = mr.AudioStreamer()
            
            self.status_label.config(text="ステータス: 待機中", fg="#333")
            self.start_btn.config(state=tk.NORMAL)
            self.log_message("初期化完了！「スタート」ボタンを押してください。")
        except Exception as e:
            self.log_message(f"初期化エラー: {e}")
            self.status_label.config(text="ステータス: エラー", fg="red")

    def log_message(self, message):
        # スレッドセーフにログを更新するための処理
        self.root.after(0, self._append_log, message)

    #ログエリアにメッセージを追加する内部関数
    def _append_log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    #現在の推奨トラックをUIに表示する関数
    def update_song_display(self, song_name):
        self.root.after(0, lambda: self.current_song_label.config(text=song_name))

    #音声録音開始
    def start_monitoring(self):
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="ステータス: リスニング中...", fg="#4caf50")
        
        # マイクを別スレッドで開始（UIをフリーズさせないため）
        self.thread = threading.Thread(target=self.run_system_loop, daemon=True)
        self.thread.start()
        
    #音声録音停止
    def stop_monitoring(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="ステータス: ⏹ 停止中", fg="#f44336")
        self.log_message("\nシステムを停止しました。")

    def run_system_loop(self):
        # 元コードの main() に相当する処理
        stream = sd.InputStream(
            samplerate=mr.Config.SR, channels=1, dtype='float32', 
            blocksize=mr.Config.CHUNK_SIZE, callback=self.streamer.callback
        )
        
        # 現在画面に表示されている推奨トラックを保持する変数
        current_playing = None
        self.log_message("-------------------------------------------------")
        self.log_message("マイク開始")
        
        #
        with stream:
            #self.is.runningは録音中かどうか
            while self.is_running:
                # 8秒分の音声を録音しつつ、UIがフリーズしないように1秒ごとにチェック
                for _ in range(8):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
                if not self.is_running:
                    break

                self.streamer.update_buffer()
                self.log_message("波形を検索中...")
                
                env_vector = self.yamnet.extract_vector(self.streamer.buffer)
                best_match, similarity = self.matcher.find_best_match(env_vector)
                
                if best_match:
                    if best_match != current_playing:
                        self.log_message(f"[マッチ成功] 推奨トラック: 『 {best_match} 』 (類似度: {similarity:.4f})")
                        self.update_song_display(best_match)
                        current_playing = best_match
                    else:
                        self.log_message(f"    (類似度: {similarity:.4f} / ムード継続: {best_match})")
                else:
                    self.log_message("(マッチする楽曲が見つかりませんでした)")
                
                self.log_message("-------------------------------------------------")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderUI(root)
    root.mainloop()