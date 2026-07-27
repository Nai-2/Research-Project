import os
import sys
import queue
import threading
import traceback
import importlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


# ============================================================
# 設定
# ============================================================

# 元の解析コードのファイル名を、拡張子なしで指定します。
# 元コードが test3.py の場合は "test3"
ENGINE_MODULE = "test3"

SUPPORTED_FILE_TYPES = [
    (
        "動画・音声ファイル",
        "*.mp4 *.mov *.avi *.mkv *.webm *.m4v "
        "*.wav *.mp3 *.m4a *.aac *.flac"
    ),
    ("動画ファイル", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
    ("音声ファイル", "*.wav *.mp3 *.m4a *.aac *.flac"),
    ("すべてのファイル", "*.*"),
]


class QueueWriter:
    """
    print()による出力を、TkinterのUIへ渡すためのクラスです。

    解析コード内のprint()を変更せずに、
    UIのログ欄へ表示できます。
    """

    def __init__(self, message_queue):
        self.message_queue = message_queue

    def write(self, text):
        if text:
            self.message_queue.put(("log", text))

    def flush(self):
        pass


class MusicSearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("環境音・動画連動 音楽検索システム")
        self.root.geometry("1050x720")
        self.root.minsize(850, 600)

        self.message_queue = queue.Queue()

        self.engine = None
        self.is_analyzing = False
        self.selected_file = tk.StringVar()
        self.status_text = tk.StringVar(value="解析エンジンを準備しています...")
        self.file_info_text = tk.StringVar(value="動画または音声ファイルを選択してください。")

        self.create_style()
        self.create_widgets()

        # キューに入ったメッセージを定期的に確認
        self.root.after(100, self.process_message_queue)

        # モデルやデータベースの読み込みは時間がかかるため、
        # UIとは別のスレッドで行います。
        self.load_engine_async()

    def create_style(self):
        style = ttk.Style()

        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=("Yu Gothic UI", 20, "bold")
        )

        style.configure(
            "Subtitle.TLabel",
            font=("Yu Gothic UI", 10)
        )

        style.configure(
            "Section.TLabelframe.Label",
            font=("Yu Gothic UI", 11, "bold")
        )

        style.configure(
            "Start.TButton",
            font=("Yu Gothic UI", 12, "bold"),
            padding=(20, 10)
        )

        style.configure(
            "Normal.TButton",
            font=("Yu Gothic UI", 10),
            padding=(12, 7)
        )

        style.configure(
            "Status.TLabel",
            font=("Yu Gothic UI", 10, "bold")
        )

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # ----------------------------------------------------
        # タイトル
        # ----------------------------------------------------
        title_label = ttk.Label(
            main_frame,
            text="🎵 環境音・動画連動 音楽検索システム",
            style="Title.TLabel"
        )
        title_label.pack(anchor="w")

        subtitle_label = ttk.Label(
            main_frame,
            text=(
                "動画・音声の雰囲気、波形の展開、キーワードを解析し、"
                "複数の評価方法で似ている楽曲を検索します。"
            ),
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(anchor="w", pady=(4, 18))

        # ----------------------------------------------------
        # ファイル選択エリア
        # ----------------------------------------------------
        file_frame = ttk.LabelFrame(
            main_frame,
            text="1．解析するファイル",
            padding=15,
            style="Section.TLabelframe"
        )
        file_frame.pack(fill="x", pady=(0, 15))

        file_entry = ttk.Entry(
            file_frame,
            textvariable=self.selected_file,
            state="readonly"
        )
        file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        select_button = ttk.Button(
            file_frame,
            text="ファイルを選択",
            command=self.select_file,
            style="Normal.TButton"
        )
        select_button.pack(side="right")

        info_label = ttk.Label(
            main_frame,
            textvariable=self.file_info_text,
            wraplength=950
        )
        info_label.pack(fill="x", pady=(0, 12))

        # ----------------------------------------------------
        # 操作エリア
        # ----------------------------------------------------
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(0, 15))

        self.start_button = ttk.Button(
            control_frame,
            text="解析を開始",
            command=self.start_analysis,
            state="disabled",
            style="Start.TButton"
        )
        self.start_button.pack(side="left")

        clear_button = ttk.Button(
            control_frame,
            text="表示をクリア",
            command=self.clear_log,
            style="Normal.TButton"
        )
        clear_button.pack(side="left", padx=(10, 0))

        self.progress_bar = ttk.Progressbar(
            control_frame,
            mode="indeterminate",
            length=260
        )
        self.progress_bar.pack(side="right", padx=(15, 0))

        status_label = ttk.Label(
            control_frame,
            textvariable=self.status_text,
            style="Status.TLabel"
        )
        status_label.pack(side="right")

        # ----------------------------------------------------
        # ログ・解析結果エリア
        # ----------------------------------------------------
        result_frame = ttk.LabelFrame(
            main_frame,
            text="2．解析状況・検索結果",
            padding=10,
            style="Section.TLabelframe"
        )
        result_frame.pack(fill="both", expand=True)

        self.log_text = ScrolledText(
            result_frame,
            wrap="word",
            font=("Consolas", 10),
            padx=10,
            pady=10,
            state="disabled"
        )
        self.log_text.pack(fill="both", expand=True)

        # ログ表示用のタグ
        self.log_text.tag_configure(
            "heading",
            font=("Yu Gothic UI", 11, "bold")
        )

        self.log_text.tag_configure(
            "error",
            foreground="#b00020"
        )

        self.log_text.tag_configure(
            "success",
            foreground="#087f23"
        )

        self.append_log(
            "解析エンジンを読み込んでいます。\n"
            "初回はYAMNetモデルの読み込みに時間がかかる場合があります。\n\n"
        )

    def load_engine_async(self):
        """
        元の解析コードをバックグラウンドで読み込みます。
        """

        self.progress_bar.start(10)

        thread = threading.Thread(
            target=self.load_engine_worker,
            daemon=True
        )
        thread.start()

    def load_engine_worker(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        writer = QueueWriter(self.message_queue)

        try:
            # test3.py内のprint()もUIへ表示する
            sys.stdout = writer
            sys.stderr = writer

            self.engine = importlib.import_module(ENGINE_MODULE)

            if not hasattr(self.engine, "execute_multi_profile_search"):
                raise AttributeError(
                    f"{ENGINE_MODULE}.py に "
                    "execute_multi_profile_search() が見つかりません。"
                )

            self.message_queue.put(("engine_ready", None))

        except Exception:
            error_message = traceback.format_exc()
            self.message_queue.put(("engine_error", error_message))

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def select_file(self):
        """
        ファイル選択ダイアログを表示します。
        """

        file_path = filedialog.askopenfilename(
            title="解析する動画または音声を選択",
            filetypes=SUPPORTED_FILE_TYPES
        )

        if not file_path:
            return

        normalized_path = os.path.normpath(file_path)
        self.selected_file.set(normalized_path)

        try:
            file_size_mb = os.path.getsize(normalized_path) / (1024 * 1024)

            self.file_info_text.set(
                f"選択中: {os.path.basename(normalized_path)} "
                f"（{file_size_mb:.1f} MB）"
            )

        except OSError:
            self.file_info_text.set(
                f"選択中: {os.path.basename(normalized_path)}"
            )

        if self.engine is not None and not self.is_analyzing:
            self.start_button.config(state="normal")

    def start_analysis(self):
        """
        選択されたファイルを解析します。
        """

        target_file = self.selected_file.get()

        if not target_file:
            messagebox.showwarning(
                "ファイル未選択",
                "解析する動画または音声ファイルを選択してください。"
            )
            return

        if not os.path.isfile(target_file):
            messagebox.showerror(
                "ファイルエラー",
                "選択したファイルが見つかりません。"
            )
            return

        if self.engine is None:
            messagebox.showinfo(
                "準備中",
                "解析エンジンの読み込みが完了していません。"
            )
            return

        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.start_button.config(state="disabled")
        self.status_text.set("解析しています...")
        self.progress_bar.start(10)

        self.append_log("\n" + "=" * 80 + "\n", tag="heading")
        self.append_log(
            f"解析対象: {target_file}\n",
            tag="heading"
        )
        self.append_log("=" * 80 + "\n\n", tag="heading")

        analysis_thread = threading.Thread(
            target=self.analysis_worker,
            args=(target_file,),
            daemon=True
        )
        analysis_thread.start()

    def analysis_worker(self, target_file):
        """
        元コードのexecute_multi_profile_search()を実行します。
        """

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        writer = QueueWriter(self.message_queue)

        try:
            # 元コードのprint()をUIのログ欄へ転送
            sys.stdout = writer
            sys.stderr = writer

            self.engine.execute_multi_profile_search(target_file)

            self.message_queue.put(("analysis_done", None))

        except Exception:
            error_message = traceback.format_exc()
            self.message_queue.put(("analysis_error", error_message))

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def process_message_queue(self):
        """
        別スレッドから送られたメッセージをUIに反映します。
        Tkinterの部品はメインスレッドからのみ操作します。
        """

        try:
            while True:
                message_type, content = self.message_queue.get_nowait()

                if message_type == "log":
                    self.append_log(content)

                elif message_type == "engine_ready":
                    self.progress_bar.stop()
                    self.status_text.set("準備完了")

                    self.append_log(
                        "\n✅ 解析エンジンの準備が完了しました。\n",
                        tag="success"
                    )
                    self.append_log(
                        "動画または音声ファイルを選択して、"
                        "「解析を開始」を押してください。\n\n"
                    )

                    if self.selected_file.get():
                        self.start_button.config(state="normal")

                elif message_type == "engine_error":
                    self.progress_bar.stop()
                    self.status_text.set("読み込みエラー")

                    self.append_log(
                        "\n❌ 解析エンジンの読み込みに失敗しました。\n",
                        tag="error"
                    )
                    self.append_log(content, tag="error")

                    messagebox.showerror(
                        "読み込みエラー",
                        f"{ENGINE_MODULE}.py の読み込みに失敗しました。\n\n"
                        "ファイル名と必要なデータベースファイルを確認してください。"
                    )

                elif message_type == "analysis_done":
                    self.is_analyzing = False
                    self.progress_bar.stop()
                    self.status_text.set("解析完了")
                    self.start_button.config(state="normal")

                    self.append_log(
                        "\n✅ すべての解析が完了しました。\n",
                        tag="success"
                    )

                elif message_type == "analysis_error":
                    self.is_analyzing = False
                    self.progress_bar.stop()
                    self.status_text.set("解析エラー")
                    self.start_button.config(state="normal")

                    self.append_log(
                        "\n❌ 解析中にエラーが発生しました。\n",
                        tag="error"
                    )
                    self.append_log(content, tag="error")

                    messagebox.showerror(
                        "解析エラー",
                        "解析中にエラーが発生しました。\n"
                        "詳細は画面内のログを確認してください。"
                    )

        except queue.Empty:
            pass

        self.root.after(100, self.process_message_queue)

    def append_log(self, text, tag=None):
        """
        ログ欄に文字列を追加します。
        """

        self.log_text.config(state="normal")

        if tag:
            self.log_text.insert("end", text, tag)
        else:
            self.log_text.insert("end", text)

        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def clear_log(self):
        """
        ログ欄の表示を消去します。
        """

        if self.is_analyzing:
            result = messagebox.askyesno(
                "表示をクリア",
                "解析中です。ログの表示だけをクリアしますか？\n"
                "解析処理自体は継続されます。"
            )

            if not result:
                return

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")


def main():
    root = tk.Tk()
    MusicSearchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()