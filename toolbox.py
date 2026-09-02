from __future__ import annotations

import base64
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scrcpy_human_automation import AdbDevice, GameStateMachine, TemplateMatcher  # noqa: E402
from scrcpy_human_automation.frame_source import ScrcpyVideoFrameSource  # noqa: E402
from scrcpy_human_automation.runner import WorkflowRunner  # noqa: E402


DARK_BG = "#0b0f14"
DARK_PANEL = "#111827"
DARK_PANEL_ALT = "#172033"
DARK_TEXT = "#e5e7eb"
DARK_MUTED = "#9ca3af"
DARK_ACCENT = "#38bdf8"


class FormDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        title: str,
        fields: list[tuple[str, str, object]],
        extra_actions: list[tuple[str, str]] | None = None,
        preview_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result: dict[str, str] | None = None
        self.entries: dict[str, ttk.Entry] = {}
        self.preview_photo: tk.PhotoImage | None = None
        self.transient(parent)
        self.grab_set()

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        row_offset = 0
        if preview_path is not None:
            preview = self._load_preview(preview_path)
            if preview is not None:
                ttk.Label(body, text=f"当前图片: {preview_path.name}").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
                ttk.Label(body, image=preview).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))
                row_offset = 2
            else:
                ttk.Label(body, text=f"当前图片未找到: {preview_path.name}", foreground="#a33").grid(
                    row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
                )
                row_offset = 1

        for row, (key, label, value) in enumerate(fields, start=row_offset):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(body, width=32)
            entry.insert(0, self._format_value(value))
            entry.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=4)
            self.entries[key] = entry
        body.columnconfigure(1, weight=1)

        actions = ttk.Frame(body)
        actions.grid(row=len(fields) + row_offset, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(actions, text="取消", command=self._cancel).pack(side="right")
        ttk.Button(actions, text="保存", command=self._ok).pack(side="right", padx=(0, 8))
        for action_id, label in reversed(extra_actions or []):
            ttk.Button(actions, text=label, command=lambda value=action_id: self._extra(value)).pack(side="right", padx=(0, 8))

        self.bind("<Return>", lambda _event: self._ok())
        self.bind("<Escape>", lambda _event: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)
        self.geometry(f"+{x}+{y}")
        first = next(iter(self.entries.values()), None)
        if first:
            first.focus_set()
            first.selection_range(0, "end")
        parent.wait_window(self)

    def _ok(self) -> None:
        self.result = {key: entry.get().strip() for key, entry in self.entries.items()}
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()

    def _extra(self, action_id: str) -> None:
        self.result = {"__action__": action_id}
        self.destroy()

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)

    def _load_preview(self, path: Path) -> tk.PhotoImage | None:
        if not path.exists():
            return None
        try:
            image_bytes = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return None
            height, width = image.shape[:2]
            scale = min(1.0, 320 / max(1, width), 120 / max(1, height))
            if scale < 1.0:
                image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            ok, png = cv2.imencode(".png", image)
            if not ok:
                return None
            encoded = base64.b64encode(png.tobytes()).decode("ascii")
            self.preview_photo = tk.PhotoImage(data=encoded)
            return self.preview_photo
        except OSError:
            return None


class ToolboxApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("手机自动化工具箱")
        self.geometry("1600x860")
        self.minsize(1300, 720)
        self.configure(bg=DARK_BG)

        self.templates_dir = ROOT / "templates"
        self.template_meta_dir = ROOT / "templates_meta"
        self.step_previews_dir = ROOT / "step_previews"
        self.flows_dir = ROOT / "flows"
        self.logs_dir = ROOT / "logs"
        self.recordings_dir = ROOT / "recordings"
        self.templates_dir.mkdir(exist_ok=True)
        self.template_meta_dir.mkdir(exist_ok=True)
        self.step_previews_dir.mkdir(exist_ok=True)
        self.flows_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.recordings_dir.mkdir(exist_ok=True)
        self.log_path = self.logs_dir / f"toolbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.flow_path: Path | None = None
        self.flow_selected_for_run = False
        self.flow_frame_source = "adb"
        self.flow_name_var = tk.StringVar(value="")
        self.zhongkui_progress_var = tk.StringVar(value="抓鬼: -")

        self.device = AdbDevice()
        self.matcher = TemplateMatcher(self.device)
        self.state_machine = GameStateMachine(self.matcher, self.templates_dir, self.matcher.frame_source)
        self.steps: list[dict] = []
        self.screen_image: np.ndarray | None = None
        self.force_landscape_var = tk.BooleanVar(value=True)
        self.preview_photo: tk.PhotoImage | None = None
        self.template_preview_photo: tk.PhotoImage | None = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.drag_start: tuple[int, int] | None = None
        self.selection_id: int | None = None
        self.selection_box: tuple[int, int, int, int] | None = None
        self.last_selection_box: tuple[int, int, int, int] | None = None
        self.coordinate_pick_mode = False
        self.coordinate_edit_index: int | None = None
        self.coordinate_quick_update = False
        self.pending_new_region_click = False
        self.pending_column_region_assignment: tuple[int, str] | None = None
        self.pending_step_region_assignment: tuple[int, str] | None = None
        self.preview_visible = False
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.recording_event = threading.Event()
        self.recording_worker: threading.Thread | None = None
        self.recording_process: subprocess.Popen | None = None
        self.recording_segment_seconds = 600
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self._configure_style()
        self._build_ui()
        self._load_flow()
        self._refresh_templates()
        self._log(f"日志文件: {self.log_path}")
        self.after(100, self._drain_events)
        self.after(250, self._check_device)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=DARK_BG, foreground=DARK_TEXT, fieldbackground=DARK_PANEL)
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabelframe", background=DARK_BG, foreground=DARK_TEXT)
        style.configure("TLabelframe.Label", background=DARK_BG, foreground=DARK_TEXT)
        style.configure("TLabel", background=DARK_BG, foreground=DARK_TEXT)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"), background=DARK_BG, foreground=DARK_TEXT)
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 10), background=DARK_BG, foreground=DARK_MUTED)
        style.configure("TButton", background=DARK_PANEL_ALT, foreground=DARK_TEXT, bordercolor=DARK_PANEL_ALT, focusthickness=1, focuscolor=DARK_ACCENT)
        style.map("TButton", background=[("active", "#22304a"), ("disabled", "#1f2937")], foreground=[("disabled", "#6b7280")])
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"), background="#0f766e", foreground="#ecfeff")
        style.map("Accent.TButton", background=[("active", "#0d9488"), ("disabled", "#1f2937")])
        style.configure("TNotebook", background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=DARK_PANEL, foreground=DARK_TEXT, padding=(10, 6))
        style.map("TNotebook.Tab", background=[("selected", DARK_PANEL_ALT)], foreground=[("selected", DARK_ACCENT)])
        style.configure("TCombobox", fieldbackground=DARK_PANEL, background=DARK_PANEL, foreground=DARK_TEXT, arrowcolor=DARK_TEXT)
        style.configure("TSpinbox", fieldbackground=DARK_PANEL, background=DARK_PANEL, foreground=DARK_TEXT)
        style.configure("TCheckbutton", background=DARK_BG, foreground=DARK_TEXT)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(16, 12, 16, 8))
        header.pack(fill="x")
        ttk.Label(header, text="手机自动化工具箱", style="Title.TLabel").pack(side="left")
        self.device_status = tk.StringVar(value="正在检查设备...")
        ttk.Label(header, textvariable=self.device_status, style="Status.TLabel").pack(side="left", padx=20)
        ttk.Button(header, text="启动 scrcpy", command=self._start_scrcpy).pack(side="right")
        ttk.Button(header, text="观察状态", command=self._observe_state_async).pack(side="right", padx=(0, 8))
        ttk.Button(header, text="刷新截图", command=self._capture_async).pack(side="right", padx=8)
        self.preview_toggle_button = ttk.Button(header, text="Show Preview", command=self._toggle_preview)
        self.preview_toggle_button.pack(side="right", padx=(0, 8))

        self.pane = ttk.Panedwindow(self, orient="horizontal")
        self.pane.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self.preview_frame = ttk.LabelFrame(self.pane, text="手机画面：拖动鼠标框选模板", padding=8)
        self.canvas = tk.Canvas(self.preview_frame, bg="#171a1f", highlightthickness=0, cursor="crosshair", height=540)
        self.canvas.pack(fill="x", expand=False)
        self.canvas.bind("<ButtonPress-1>", self._selection_start)
        self.canvas.bind("<B1-Motion>", self._selection_drag)
        self.canvas.bind("<ButtonRelease-1>", self._selection_end)
        self.canvas.bind("<Configure>", lambda _event: self._render_preview())
        preview_actions = ttk.Frame(self.preview_frame)
        preview_actions.pack(fill="x", pady=(8, 0))
        ttk.Button(preview_actions, text="将选区保存为模板", command=self._save_selection).pack(side="left")
        ttk.Checkbutton(preview_actions, text="强制横屏", variable=self.force_landscape_var, command=self._apply_orientation_and_render).pack(side="left", padx=(8, 0))
        self.selection_text = tk.StringVar(value="请先刷新截图")
        ttk.Label(preview_actions, textvariable=self.selection_text).pack(side="left", padx=10)

        right = ttk.Frame(self.pane)
        self.pane.add(right, weight=2)
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        template_tab = ttk.Frame(notebook, padding=10)
        flow_tab = ttk.Frame(notebook, padding=10)
        notebook.add(template_tab, text="模板库")
        notebook.add(flow_tab, text="流程编排")
        self._build_template_tab(template_tab)
        self._build_flow_tab(flow_tab)
        notebook.select(flow_tab)

        log_frame = ttk.LabelFrame(self, text="运行日志", padding=6)
        log_frame.pack(fill="x", padx=16, pady=(0, 14))
        self.log_text = tk.Text(
            log_frame,
            height=7,
            state="disabled",
            bg=DARK_PANEL,
            fg=DARK_TEXT,
            insertbackground=DARK_TEXT,
            selectbackground="#1d4ed8",
            relief="flat",
        )
        self.log_text.pack(fill="x")

    def _toggle_preview(self) -> None:
        if self.preview_visible:
            self._hide_preview()
        else:
            self._show_preview()

    def _show_preview(self) -> None:
        if self.preview_visible:
            return
        panes = set(self.pane.panes())
        if str(self.preview_frame) not in panes:
            self.pane.insert(0, self.preview_frame, weight=5)
        self.preview_visible = True
        self.preview_toggle_button.configure(text="Hide Preview")
        self.after(50, self._render_preview)

    def _hide_preview(self) -> None:
        if not self.preview_visible:
            return
        try:
            self.pane.forget(self.preview_frame)
        except tk.TclError:
            pass
        self.preview_visible = False
        self.preview_toggle_button.configure(text="Show Preview")

    def _build_template_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="你添加的模板图片").pack(anchor="w")
        content = ttk.Frame(parent)
        content.pack(fill="both", expand=True, pady=8)
        self.template_list = tk.Listbox(
            content,
            height=18,
            exportselection=False,
            bg=DARK_PANEL,
            fg=DARK_TEXT,
            selectbackground="#1d4ed8",
            selectforeground="#ffffff",
            highlightthickness=0,
            relief="flat",
        )
        self.template_list.pack(side="left", fill="both", expand=True)
        self.template_list.bind("<<ListboxSelect>>", self._show_selected_template_preview)
        self.template_list.bind("<ButtonRelease-1>", self._show_selected_template_preview)
        self.template_list.bind("<KeyRelease-Up>", self._show_selected_template_preview)
        self.template_list.bind("<KeyRelease-Down>", self._show_selected_template_preview)
        preview_box = ttk.LabelFrame(content, text="模板预览", padding=8)
        preview_box.pack(side="left", fill="both", padx=(10, 0))
        self.template_preview_label = ttk.Label(preview_box, text="点击模板名称查看图片", anchor="center")
        self.template_preview_label.pack(fill="both", expand=True)
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Button(row, text="导入图片", command=self._import_template).pack(side="left")
        ttk.Button(row, text="删除", command=self._delete_template).pack(side="left", padx=6)
        ttk.Button(row, text="测试识别", command=self._test_template_async).pack(side="left")
        ttk.Label(
            parent,
            text="建议模板只截取按钮或图标的稳定区域，避免时间、头像等会变化的内容。",
            wraplength=360,
            foreground=DARK_MUTED,
        ).pack(anchor="w", pady=(12, 0))

    def _build_flow_tab(self, parent: ttk.Frame) -> None:
        profile_row = ttk.Frame(parent)
        profile_row.pack(fill="x", pady=(0, 8))
        ttk.Label(profile_row, text="当前流程").pack(side="left")
        self.flow_selector = ttk.Combobox(
            profile_row,
            textvariable=self.flow_name_var,
            state="readonly",
            width=18,
        )
        self.flow_selector.pack(side="left", padx=6, fill="x", expand=True)
        self.flow_selector.bind("<<ComboboxSelected>>", self._select_flow)
        ttk.Button(profile_row, text="新建", command=self._new_flow).pack(side="left")

        self.step_list = tk.Listbox(
            parent,
            height=14,
            exportselection=False,
            bg=DARK_PANEL,
            fg=DARK_TEXT,
            selectbackground="#1d4ed8",
            selectforeground="#ffffff",
            highlightthickness=0,
            relief="flat",
        )
        self.step_list.pack(fill="both", expand=True)
        self.step_list.bind("<Double-Button-1>", lambda _event: self._edit_step())
        add_row = ttk.Frame(parent)
        add_row.pack(fill="x", pady=8)
        ttk.Button(add_row, text="+ 识别点击", command=self._add_click_step).pack(side="left")
        ttk.Button(add_row, text="+ 范围点击", command=self._add_region_click_step).pack(side="left", padx=5)
        ttk.Button(add_row, text="+ 随机等待", command=self._add_delay_step).pack(side="left", padx=5)
        ttk.Button(add_row, text="+ 等模板点击", command=self._add_wait_template_click_step).pack(side="left", padx=5)
        ttk.Button(add_row, text="+ 模板滑动重试", command=self._add_template_scroll_click_step).pack(side="left", padx=5)
        ttk.Button(add_row, text="+ 按列范围点击", command=self._add_template_column_region_click_step).pack(side="left", padx=5)
        ttk.Button(add_row, text="+ 等区域稳定", command=self._add_wait_region_still_step).pack(side="left", padx=5)
        ttk.Button(add_row, text="+ 等战斗结束", command=self._add_wait_battle_complete_step).pack(side="left", padx=5)
        ttk.Button(add_row, text="+ 自然滑动", command=self._add_swipe_step).pack(side="left")

        edit_row = ttk.Frame(parent)
        edit_row.pack(fill="x")
        ttk.Button(edit_row, text="上移", command=lambda: self._move_step(-1)).pack(side="left")
        ttk.Button(edit_row, text="下移", command=lambda: self._move_step(1)).pack(side="left", padx=5)
        ttk.Button(edit_row, text="修改步骤", command=self._edit_step).pack(side="left", padx=(0, 5))
        ttk.Button(edit_row, text="删除步骤", command=self._delete_step).pack(side="left")

        run_row = ttk.Frame(parent)
        run_row.pack(fill="x", pady=(16, 0))
        ttk.Label(run_row, text="循环次数(0=无限)").pack(side="left")
        self.repeat_var = tk.IntVar(value=0)
        ttk.Spinbox(run_row, from_=0, to=999, width=6, textvariable=self.repeat_var).pack(side="left", padx=6)
        self.run_button = ttk.Button(run_row, text="Run All", style="Accent.TButton", command=self._run_flow, state="disabled")
        self.run_button.pack(side="left", padx=(8, 5))
        self.run_from_selected_button = ttk.Button(run_row, text="Run Selected", command=self._run_flow_from_selected, state="disabled")
        self.run_from_selected_button.pack(side="left", padx=(0, 5))
        ttk.Button(run_row, text="设置抓鬼计数", command=self._set_zhongkui_progress).pack(side="left", padx=(0, 5))
        ttk.Label(run_row, textvariable=self.zhongkui_progress_var).pack(side="left", padx=(0, 5))
        self.stop_button = ttk.Button(run_row, text="Stop", command=self._stop_flow, state="disabled")
        self.stop_button.pack(side="left")
        self.record_button = ttk.Button(run_row, text="Start Rec", command=self._toggle_recording)
        self.record_button.pack(side="left", padx=(8, 0))
        self._refresh_zhongkui_progress_label()

    def _set_zhongkui_progress(self) -> None:
        key = "zhongkui_ghost"
        path = ROOT / "state" / f"{key}.json"
        current = self._read_zhongkui_progress()

        dialog = tk.Toplevel(self)
        dialog.title("设置抓鬼计数")
        dialog.configure(bg=DARK_BG)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.lift()
        dialog.attributes("-topmost", True)
        dialog.after(250, lambda: dialog.attributes("-topmost", False))

        body = ttk.Frame(dialog, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="今天已完成抓鬼数量").pack(anchor="w")
        ttk.Label(body, text="未满20填实际数量；超过20直接填20。").pack(anchor="w", pady=(4, 10))
        value_var = tk.IntVar(value=current)
        entry = ttk.Entry(body, textvariable=value_var, width=10)
        entry.pack(anchor="w")
        entry.focus_set()
        entry.select_range(0, "end")

        def save(value: int | None = None) -> None:
            try:
                completed = int(value if value is not None else value_var.get())
            except (TypeError, ValueError, tk.TclError):
                messagebox.showerror("数值无效", "请输入 0-999 的数字。", parent=dialog)
                return
            completed = max(0, min(999, completed))
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "task": key,
                "completed": completed,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._refresh_zhongkui_progress_label()
            self._log(f"已设置抓鬼计数: {completed}")
            dialog.destroy()

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(12, 0))
        ttk.Button(actions, text="设为0", command=lambda: save(0)).pack(side="left")
        ttk.Button(actions, text="设为20", command=lambda: save(20)).pack(side="left", padx=6)
        ttk.Button(actions, text="保存", command=save).pack(side="right")
        ttk.Button(actions, text="取消", command=dialog.destroy).pack(side="right", padx=(0, 6))
        dialog.bind("<Return>", lambda _event: save())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())

    def _read_zhongkui_progress(self) -> int:
        path = ROOT / "state" / "zhongkui_ghost.json"
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if str(data.get("date", "")) != datetime.now().strftime("%Y-%m-%d"):
                return 0
            return max(0, int(data.get("completed", 0)))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0

    def _refresh_zhongkui_progress_label(self) -> None:
        self.zhongkui_progress_var.set(f"抓鬼: {self._read_zhongkui_progress()}")

    def _check_device(self) -> None:
        def task() -> None:
            try:
                self.device.ensure_connected()
                size = self.device.screen_size(refresh=True)
                self.events.put(("status", f"设备已连接 · {size.width} × {size.height}"))
                self.events.put(("log", "ADB 设备连接正常"))
                self.events.put(("capture", None))
            except Exception as exc:
                self.events.put(("status", "未发现 ADB 设备"))
                self.events.put(("log", f"设备检查失败: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _capture_async(self) -> None:
        self.selection_box = None
        self.last_selection_box = None
        threading.Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self) -> None:
        try:
            image = self._normalize_screenshot_orientation(self.matcher.screenshot())
            self.events.put(("image", image))
            image_h, image_w = image.shape[:2]
            self.events.put(("log", f"???????: {image_w}x{image_h}"))
        except Exception as exc:
            self.events.put(("error", f"????: {exc}"))

    def _observe_state_async(self) -> None:
        def task() -> None:
            try:
                state = self.state_machine.observe()
                signals = ", ".join(f"{signal.name}:{signal.confidence:.3f}" for signal in state.signals) or "none"
                self.events.put(("log", f"state={state.name} confidence={state.confidence:.3f} signals={signals}"))
            except Exception as exc:
                self.events.put(("error", f"观察状态失败: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _capture_sync(self) -> bool:
        try:
            self.selection_box = None
            self.last_selection_box = None
            self.screen_image = self._normalize_screenshot_orientation(self.matcher.screenshot())
            image_h, image_w = self.screen_image.shape[:2]
            self.selection_text.set(f"????? {image_w}x{image_h}????????????")
            self._render_preview()
            self._log(f"???????: {image_w}x{image_h}")
            return True
        except Exception as exc:
            messagebox.showerror("????", f"?????????{exc}")
            return False

    def _normalize_screenshot_orientation(self, image: np.ndarray) -> np.ndarray:
        if self.force_landscape_var.get() and image.shape[0] > image.shape[1]:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

    def _apply_orientation_and_render(self) -> None:
        if self.screen_image is not None:
            self.screen_image = self._normalize_screenshot_orientation(self.screen_image)
            image_h, image_w = self.screen_image.shape[:2]
            self.selection_text.set(f"???? {image_w}x{image_h}")
        self._render_preview()

    def _render_preview(self) -> None:
        if self.screen_image is None or self.canvas.winfo_width() < 10:
            return
        image_h, image_w = self.screen_image.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        self.preview_scale = min(canvas_w / image_w, canvas_h / image_h)
        width = max(1, int(image_w * self.preview_scale))
        height = max(1, int(image_h * self.preview_scale))
        resized = cv2.resize(self.screen_image, (width, height), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", resized)
        if not ok:
            return
        encoded = base64.b64encode(png.tobytes()).decode("ascii")
        self.preview_photo = tk.PhotoImage(data=encoded)
        self.preview_offset = ((canvas_w - width) // 2, (canvas_h - height) // 2)
        self.canvas.delete("all")
        self.canvas.create_image(*self.preview_offset, anchor="nw", image=self.preview_photo)
        self.selection_id = None

    def _selection_start(self, event: tk.Event) -> None:
        if self.screen_image is None:
            return
        self.drag_start = (event.x, event.y)
        if self.selection_id:
            self.canvas.delete(self.selection_id)
        self.selection_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#ffbf3f", width=1)

    def _selection_drag(self, event: tk.Event) -> None:
        if self.drag_start and self.selection_id:
            self.canvas.coords(self.selection_id, *self.drag_start, event.x, event.y)

    def _selection_end(self, event: tk.Event) -> None:
        if not self.drag_start or self.screen_image is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        ox, oy = self.preview_offset
        image_h, image_w = self.screen_image.shape[:2]
        left = max(0, min(image_w, int((min(x1, x2) - ox) / self.preview_scale)))
        top = max(0, min(image_h, int((min(y1, y2) - oy) / self.preview_scale)))
        right = max(0, min(image_w, int((max(x1, x2) - ox) / self.preview_scale)))
        bottom = max(0, min(image_h, int((max(y1, y2) - oy) / self.preview_scale)))
        self.selection_box = (left, top, right, bottom) if right - left >= 8 and bottom - top >= 8 else None
        if self.selection_box:
            self.last_selection_box = self.selection_box
        self.selection_text.set(
            f"选区 {right - left} × {bottom - top}" if self.selection_box else "选区太小，请重新框选"
        )
        self.drag_start = None
        if self.selection_box and self.pending_column_region_assignment is not None:
            edit_index, side = self.pending_column_region_assignment
            self.pending_column_region_assignment = None
            self._set_column_click_region_from_selection(edit_index, side)
        if self.selection_box and self.pending_step_region_assignment is not None:
            edit_index, key = self.pending_step_region_assignment
            self.pending_step_region_assignment = None
            self._set_step_region_from_selection(edit_index, key)
        if self.selection_box and self.pending_new_region_click:
            self.pending_new_region_click = False
            self._create_region_click_from_selection()

    def _selection_relative_region(self) -> list[float] | None:
        if self.screen_image is None:
            return None
        box = self.selection_box or self.last_selection_box
        if not box:
            return None
        image_h, image_w = self.screen_image.shape[:2]
        left, top, right, bottom = box
        return [
            round(left / image_w, 6),
            round(top / image_h, 6),
            round(right / image_w, 6),
            round(bottom / image_h, 6),
        ]

    def _set_column_click_region_from_selection(self, edit_index: int, side: str) -> bool:
        region = self._selection_relative_region()
        if region is None or not (0 <= edit_index < len(self.steps)):
            return False
        step = self.steps[edit_index]
        regions = step.get("column_click_regions")
        if not isinstance(regions, dict):
            regions = {}
        new_step = {**step}
        new_regions = dict(regions)
        new_regions[side] = region
        new_step["column_click_regions"] = new_regions
        self.steps[edit_index] = new_step
        self._steps_changed(select=edit_index)
        self.selection_text.set("已更新左列范围" if side == "left" else "已更新右列范围")
        self._log(f"已更新{'左' if side == 'left' else '右'}列点击范围: {region}")
        return True

    def _set_step_region_from_selection(self, edit_index: int, key: str) -> bool:
        region = self._selection_relative_region()
        if region is None or not (0 <= edit_index < len(self.steps)):
            return False
        new_step = {**self.steps[edit_index], key: region}
        new_step.pop("disabled", None)
        if key == "region" and new_step.get("type") in {"region_click", "conditional_region_click"}:
            image_h, image_w = self.screen_image.shape[:2]
            left, top, right, bottom = region
            center_x = int((float(left) + float(right)) * 0.5 * image_w)
            center_y = int((float(top) + float(bottom)) * 0.5 * image_h)
            new_step["use_click_region"] = True
            new_step["pixel_point"] = [center_x, center_y]
            new_step["screen_size"] = [image_w, image_h]
            new_step["point"] = [center_x / image_w, center_y / image_h]
            new_step["drift_px"] = 0
            new_step["padding_ratio"] = 0.0
            preview_path = self._save_step_region_preview(edit_index, (int(left * image_w), int(top * image_h), int(right * image_w), int(bottom * image_h)))
            if preview_path is not None:
                new_step["region_preview"] = preview_path.name
        self.steps[edit_index] = new_step
        self._steps_changed(select=edit_index)
        label = "识别区域" if key == "search_region" else "点击范围"
        self.selection_text.set(f"已更新{label}")
        self._log(f"已更新{label}: {region}")
        return True

    def _add_region_click_step(self) -> None:
        if self.screen_image is None and not self._capture_sync():
            return
        self.pending_new_region_click = True
        self._show_preview()
        self.selection_text.set("请在左侧截图框选点击范围")
        self._log("等待框选新增范围点击")

    def _create_region_click_from_selection(self) -> None:
        if self.screen_image is None or not self.selection_box:
            return
        image_h, image_w = self.screen_image.shape[:2]
        left, top, right, bottom = self.selection_box
        center_x = int((left + right) * 0.5)
        center_y = int((top + bottom) * 0.5)
        step = {
            "type": "region_click",
            "label": "范围点击",
            "pixel_point": [center_x, center_y],
            "screen_size": [image_w, image_h],
            "point": [center_x / image_w, center_y / image_h],
            "drift_px": 0,
            "use_click_region": True,
            "region": [left / image_w, top / image_h, right / image_w, bottom / image_h],
            "padding_ratio": 0.0,
            "pre_delay": [0.15, 0.45],
            "post_delay": [1.5, 2.1],
        }
        self.steps.append(step)
        edit_index = len(self.steps) - 1
        preview_path = self._save_step_region_preview(edit_index, self.selection_box)
        if preview_path is not None:
            self.steps[edit_index]["region_preview"] = preview_path.name
        self._steps_changed(select=edit_index)
        self._edit_region_click_step(edit_index)

    def _save_step_region_preview(self, edit_index: int, box: tuple[int, int, int, int]) -> Path | None:
        if self.screen_image is None:
            return None
        raw_flow_name = self.flow_path.stem if self.flow_path is not None else "unselected_flow"
        flow_name = "".join(char for char in raw_flow_name if char not in '\\/:*?"<>|').strip() or "flow"
        step = self.steps[edit_index] if 0 <= edit_index < len(self.steps) else {}
        label = "".join(char for char in str(step.get("label", "step")) if char not in '\\/:*?"<>|').strip() or "step"
        path = self.step_previews_dir / f"{flow_name}_{edit_index + 1:02d}_{label}.png"
        if self._write_template_crop(path, box):
            return path
        return None

    @staticmethod
    def _format_region(values: object) -> str:
        if not isinstance(values, list) or len(values) != 4:
            return ""
        return ",".join(f"{float(value):.6f}".rstrip("0").rstrip(".") for value in values)

    @staticmethod
    def _parse_region_text(text: str) -> list[float] | None:
        try:
            values = [float(value.strip()) for value in text.split(",")]
        except ValueError:
            return None
        if len(values) != 4:
            return None
        left, top, right, bottom = values
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            return None
        return values

    def _begin_coordinate_pick(self, edit_index: int | None = None, quick_update: bool = False) -> None:
        messagebox.showinfo("已简化", "现在统一使用“范围点击”：请点击“+ 范围点击”或在修改步骤里选择“框选设为点击范围”。")

    def _finish_coordinate_pick(self, canvas_x: int, canvas_y: int) -> None:
        self.coordinate_pick_mode = False
        self.coordinate_edit_index = None
        self.coordinate_quick_update = False

    def _quick_update_coordinate_step(self, edit_index: int, image_x: int, image_y: int) -> None:
        messagebox.showinfo("已简化", "现在统一使用“范围点击”，不再支持单点坐标更新。")

    def _configure_coordinate_step(self, image_x: int, image_y: int, edit_index: int | None) -> None:
        messagebox.showinfo("已简化", "现在统一使用“范围点击”，不再支持单点坐标步骤。")

    def _ask_step_interval(self, step: dict) -> tuple[float, float] | None:
        pre_delay = step.get("pre_delay", [1.5, 2.1])
        base_default = float(pre_delay[0]) if len(pre_delay) == 2 else 1.5
        extra_default = round(max(0.0, float(pre_delay[1]) - base_default), 3) if len(pre_delay) == 2 else 0.6
        base_delay = simpledialog.askfloat(
            "步骤间隔",
            "动作完成后的基础等待秒数（默认 1.5，可直接修改）:",
            initialvalue=base_default,
            minvalue=0.0,
            maxvalue=300.0,
            parent=self,
        )
        if base_delay is None:
            return None
        random_extra = simpledialog.askfloat(
            "随机时间",
            "在基础等待上额外随机增加的秒数（默认 0.6）:",
            initialvalue=extra_default,
            minvalue=0.0,
            maxvalue=60.0,
            parent=self,
        )
        if random_extra is None:
            return None
        return base_delay, random_extra

    def _save_selection(self) -> None:
        if self.screen_image is None:
            try:
                self.screen_image = self.matcher.screenshot()
                self._render_preview()
                self._log("保存模板前自动刷新了一次截图")
            except Exception as exc:
                messagebox.showerror("截图失败", f"无法获取手机截图：{exc}")
                return
        if not self.selection_box:
            messagebox.showinfo("提示", "截图已经有了，请在左侧手机画面上按住鼠标拖动，框选要保存为模板的按钮区域。")
            return
        name = simpledialog.askstring("保存模板", "模板名称（例如：开始按钮）:", parent=self)
        if not name:
            return
        safe_name = "".join(char for char in name.strip() if char not in '\\/:*?"<>|').strip()
        if not safe_name:
            messagebox.showerror("名称无效", "请输入有效的模板名称。")
            return
        left, top, right, bottom = self.selection_box
        path = self.templates_dir / f"{safe_name}.png"
        if path.exists() and not messagebox.askyesno("覆盖模板", f"{path.name} 已存在，是否覆盖？"):
            return
        if not self._write_template_crop(path, (left, top, right, bottom)):
            return
        self._write_template_meta(path.name, (left, top, right, bottom))
        self._refresh_templates(select=path.name)
        self._log(f"已保存模板: {path.name}")

    def _write_template_crop(self, path: Path, box: tuple[int, int, int, int]) -> bool:
        if self.screen_image is None:
            messagebox.showinfo("提示", "请先刷新截图。")
            return False
        left, top, right, bottom = box
        crop = self.screen_image[top:bottom, left:right]
        encoded_ok, encoded = cv2.imencode(".png", crop)
        if not encoded_ok:
            messagebox.showerror("保存失败", "无法将选区编码为 PNG 图片。")
            return False
        try:
            path.write_bytes(encoded.tobytes())
        except OSError as exc:
            messagebox.showerror("保存失败", f"无法写入模板图片：{exc}")
            return False
        return True

    def _overwrite_template_from_selection(self, template_name: str) -> bool:
        if self.screen_image is None:
            self._capture_sync()
        if not self.selection_box:
            messagebox.showinfo("提示", "请先在左侧截图里拖动框选要保存为模板的区域，再点修改步骤里的覆盖模板。")
            return False
        if not template_name:
            messagebox.showerror("模板名无效", "当前步骤没有模板文件名。")
            return False
        if "." not in Path(template_name).name:
            template_name = f"{template_name}.png"
        safe_name = "".join(char for char in template_name.strip() if char not in '\\/:*?"<>|').strip()
        if not safe_name:
            messagebox.showerror("模板名无效", "模板文件名不能为空。")
            return False
        path = self.templates_dir / safe_name
        if not self._write_template_crop(path, self.selection_box):
            return False
        self._write_template_meta(path.name, self.selection_box)
        self._refresh_templates(select=path.name)
        self._log(f"已用当前框选覆盖模板: {path.name}")
        messagebox.showinfo("模板已更新", f"已覆盖模板：{path.name}")
        return True

    def _template_meta_path(self, template_name: str) -> Path:
        return self.template_meta_dir / f"{Path(template_name).stem}.json"

    def _write_template_meta(self, template_name: str, box: tuple[int, int, int, int]) -> None:
        if self.screen_image is None:
            return
        image_h, image_w = self.screen_image.shape[:2]
        left, top, right, bottom = box
        data = {
            "template": template_name,
            "source_screen_size": [image_w, image_h],
            "pixel_box": [left, top, right, bottom],
            "region": [
                round(left / image_w, 6),
                round(top / image_h, 6),
                round(right / image_w, 6),
                round(bottom / image_h, 6),
            ],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            self._template_meta_path(template_name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            self._log(f"模板范围元数据写入失败: {exc}")

    def _read_template_meta(self, template_name: str) -> dict | None:
        path = self._template_meta_path(template_name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None
        region = data.get("region")
        if not isinstance(region, list) or len(region) != 4:
            return None
        return data

    def _import_template(self) -> None:
        source = filedialog.askopenfilename(
            title="选择模板图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
        )
        if not source:
            return
        source_path = Path(source)
        destination = self.templates_dir / source_path.name
        if destination.exists() and not messagebox.askyesno("覆盖模板", f"{destination.name} 已存在，是否覆盖？"):
            return
        shutil.copy2(source_path, destination)
        self._refresh_templates(select=destination.name)
        self._log(f"已导入模板: {destination.name}")

    def _delete_template(self) -> None:
        name = self._selected_template()
        if not name or not messagebox.askyesno("删除模板", f"确定删除 {name}？"):
            return
        (self.templates_dir / name).unlink(missing_ok=True)
        self._template_meta_path(name).unlink(missing_ok=True)
        self._refresh_templates()
        self._log(f"已删除模板: {name}")

    def _test_template_async(self) -> None:
        name = self._selected_template()
        if not name:
            messagebox.showinfo("提示", "请先选择一个模板。")
            return

        def task() -> None:
            try:
                match = self.matcher.find_template(self.templates_dir / name, threshold=0.0)
                confidence = match.confidence if match else 0.0
                self.events.put(("info", f"{name}\n当前画面最高匹配度：{confidence:.3f}"))
            except Exception as exc:
                self.events.put(("error", f"测试识别失败: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _add_click_step(self) -> None:
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先在模板库中导入图片，或从左侧手机截图框选保存。")
            return
        selected = self._selected_template() or templates[0]
        name = simpledialog.askstring("识别并点击", "输入模板文件名:", initialvalue=selected, parent=self)
        if not name or name not in templates:
            if name:
                messagebox.showerror("模板不存在", "请填写模板库中已有的完整文件名。")
            return
        threshold = simpledialog.askfloat("识别阈值", "匹配阈值 0~1:", initialvalue=0.88, minvalue=0.0, maxvalue=1.0, parent=self)
        if threshold is None:
            return
        timeout = simpledialog.askfloat("等待时间", "最长等待识别秒数:", initialvalue=10.0, minvalue=0.1, parent=self)
        if timeout is None:
            return
        self.steps.append({"type": "template_click", "template": name, "threshold": threshold, "timeout": timeout})
        self._steps_changed()

    def _add_wait_template_click_step(self) -> None:
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先在模板库导入图片，或从左侧手机截图框选保存。")
            return
        selected = self._selected_template() or templates[0]
        name = simpledialog.askstring("等待模板并点击", "模板文件名:", initialvalue=selected, parent=self)
        if not name or name not in templates:
            if name:
                messagebox.showerror("模板不存在", "请填写模板库中已有的完整文件名。")
            return
        threshold = simpledialog.askfloat("识别阈值", "匹配阈值 0~1:", initialvalue=0.88, minvalue=0.0, maxvalue=1.0, parent=self)
        if threshold is None:
            return
        initial_wait = simpledialog.askfloat("初始等待", "先等待多少秒再开始识别:", initialvalue=30.0, minvalue=0.0, maxvalue=600.0, parent=self)
        if initial_wait is None:
            return
        timeout = simpledialog.askfloat("最长等待", "开始识别后最多等待多少秒:", initialvalue=60.0, minvalue=0.1, maxvalue=1800.0, parent=self)
        if timeout is None:
            return
        interval = simpledialog.askfloat("轮询间隔", "每隔多少秒截图识别一次:", initialvalue=2.0, minvalue=0.2, maxvalue=60.0, parent=self)
        if interval is None:
            return
        self.steps.append(
            {
                "type": "wait_template_click",
                "template": name,
                "threshold": threshold,
                "initial_delay": [initial_wait, initial_wait + 3.0],
                "timeout": timeout,
                "interval": interval,
                "on_timeout": "pause",
                "screenshot_on_timeout": True,
            }
        )
        self._steps_changed()

    def _add_template_scroll_click_step(self) -> None:
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先在模板库导入图片，或从左侧手机截图框选保存。")
            return
        selected = self._selected_template() or templates[0]
        dialog = FormDialog(
            self,
            "新增模板滑动重试",
            [
                ("template", "目标模板文件名", selected),
                ("threshold", "匹配阈值 0~1", 0.86),
                ("attempts", "最多尝试次数", 2),
                ("timeout_per_attempt", "每次最多识别秒", 3),
                ("interval", "轮询间隔秒", 0.5),
                ("click_offset_x", "点击偏移 X 像素", 360),
                ("click_offset_y", "点击偏移 Y 像素", 0),
                ("drift", "点击漂移半径 px", 18),
                ("pre_swipes", "识别前先滑次数", 0),
                ("swipe_direction", "失败后滑动方向", "up"),
                ("on_timeout", "超时策略", "pause"),
            ],
            extra_actions=[("refresh", "刷新截图"), ("overwrite_template", "用框选覆盖模板")],
            preview_path=self.templates_dir / selected,
        )
        if dialog.result is None:
            return
        if dialog.result.get("__action__") == "refresh":
            if self._capture_sync():
                self._add_template_scroll_click_step()
            return
        if dialog.result.get("__action__") == "overwrite_template":
            self._overwrite_template_from_selection(selected)
            return
        step = self._parse_template_scroll_click_form(dialog.result, templates)
        if step is None:
            return
        self.steps.append(step)
        self._steps_changed()

    def _add_template_column_region_click_step(self) -> None:
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先在模板库导入图片，或从左侧手机截图框选保存。")
            return
        selected = self._selected_template() or templates[0]
        step = {
            "type": "template_column_region_click",
            "template": selected,
            "threshold": 0.84,
            "attempts": 2,
            "timeout_per_attempt": 3.0,
            "interval": 0.5,
            "pre_swipes": 0,
            "lock_column": False,
            "lock_key": selected,
            "column_click_regions": {},
            "padding_ratio": 0.18,
            "pre_delay": [0.12, 0.35],
            "post_delay": [1.5, 2.3],
            "on_timeout": "pause",
            "screenshot_on_timeout": True,
            "debug_screenshot_after": True,
        }
        self.steps.append(step)
        self._steps_changed(select=len(self.steps) - 1)
        self._edit_template_column_region_click_step(len(self.steps) - 1)

    def _add_wait_region_still_step(self) -> None:
        if self.screen_image is None:
            messagebox.showinfo("提示", "请先刷新截图，再框选要观察是否变化的区域。")
            return
        if not self.selection_box:
            messagebox.showinfo("提示", "请先在左侧截图里框选一个区域，比如任务栏、坐标区或主画面区域。")
            return
        image_h, image_w = self.screen_image.shape[:2]
        left, top, right, bottom = self.selection_box
        label = simpledialog.askstring("等待区域稳定", "步骤名称:", initialvalue="等待自动寻路/战斗结束", parent=self)
        if not label:
            return
        timeout = simpledialog.askfloat("最长等待", "最多等待多少秒:", initialvalue=480.0, minvalue=1.0, maxvalue=3600.0, parent=self)
        if timeout is None:
            return
        interval = simpledialog.askfloat("检测间隔", "每隔多少秒截图比较一次:", initialvalue=5.0, minvalue=1.0, maxvalue=60.0, parent=self)
        if interval is None:
            return
        stable_count = simpledialog.askinteger("稳定确认", "连续稳定几次才算结束:", initialvalue=3, minvalue=1, maxvalue=20, parent=self)
        if stable_count is None:
            return
        self.steps.append(
            {
                "type": "wait_region_still",
                "label": label.strip(),
                "region": [left / image_w, top / image_h, right / image_w, bottom / image_h],
                "initial_delay": [5.0, 8.0],
                "interval": interval,
                "stable_count": stable_count,
                "diff_threshold": 4.0,
                "timeout": timeout,
                "on_timeout": "pause",
                "screenshot_on_timeout": True,
            }
        )
        self._steps_changed()

    def _add_wait_battle_complete_step(self) -> None:
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先保存顶部“回合”文字模板，例如：战斗回合标志。")
            return
        selected = self._selected_template() or templates[0]
        name = simpledialog.askstring("等待战斗结束", "战斗标志模板文件名:", initialvalue=selected, parent=self)
        if name and "." not in Path(name).name:
            name = f"{name}.png"
        if not name or name not in templates:
            if name:
                messagebox.showerror("模板不存在", "请填写模板库中已有的完整文件名。")
            return
        threshold = simpledialog.askfloat("等待战斗结束", "匹配阈值 0~1:", initialvalue=0.86, minvalue=0.0, maxvalue=1.0, parent=self)
        if threshold is None:
            return
        enter_timeout = simpledialog.askfloat("等待战斗结束", "寻路/等待进入战斗最多秒数:", initialvalue=240.0, minvalue=1.0, maxvalue=1800.0, parent=self)
        if enter_timeout is None:
            return
        battle_timeout = simpledialog.askfloat("等待战斗结束", "进入战斗后最多等待秒数:", initialvalue=180.0, minvalue=1.0, maxvalue=1800.0, parent=self)
        if battle_timeout is None:
            return
        interval = simpledialog.askfloat("等待战斗结束", "每隔多少秒识别一次:", initialvalue=5.0, minvalue=1.0, maxvalue=60.0, parent=self)
        if interval is None:
            return
        self.steps.append(
            {
                "type": "wait_battle_complete",
                "template": name,
                "threshold": threshold,
                "search_region": [0.32, 0.0, 0.68, 0.18],
                "enter_timeout": enter_timeout,
                "battle_timeout": battle_timeout,
                "interval": interval,
                "confirm_gone": 3,
                "on_timeout": "pause",
                "screenshot_on_timeout": True,
                "post_delay": [2.0, 4.0],
            }
        )
        self._steps_changed()

    def _add_delay_step(self) -> None:
        minimum = simpledialog.askfloat("随机等待", "最短秒数:", initialvalue=0.5, minvalue=0.0, parent=self)
        if minimum is None:
            return
        maximum = simpledialog.askfloat("随机等待", "最长秒数:", initialvalue=1.2, minvalue=minimum, parent=self)
        if maximum is None:
            return
        self.steps.append({"type": "delay", "min": minimum, "max": maximum})
        self._steps_changed()

    def _add_swipe_step(self) -> None:
        direction = simpledialog.askstring("自然滑动", "方向: up / down / left / right", initialvalue="up", parent=self)
        if not direction:
            return
        direction = direction.strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            messagebox.showerror("方向无效", "请输入 up、down、left 或 right。")
            return
        self.steps.append({"type": "swipe", "direction": direction, "duration_min": 650, "duration_max": 1100})
        self._steps_changed()

    def _move_step(self, delta: int) -> None:
        selection = self.step_list.curselection()
        if not selection:
            return
        index = selection[0]
        target = index + delta
        if not 0 <= target < len(self.steps):
            return
        self.steps[index], self.steps[target] = self.steps[target], self.steps[index]
        self._steps_changed(select=target)

    def _delete_step(self) -> None:
        selection = self.step_list.curselection()
        if selection:
            del self.steps[selection[0]]
            self._steps_changed()

    def _edit_step(self) -> None:
        selection = self.step_list.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先选择要修改的步骤。")
            return
        index = selection[0]
        step = self.steps[index]
        action = step.get("type")

        if action in {"region_click", "conditional_region_click"}:
            self._edit_region_click_step(index)
            return

        if action == "delay":
            self._edit_delay_step(index)
            return

        if action in {"template_click", "wait_template", "wait_template_click", "wait_template_gone"}:
            self._edit_template_step(index)
            return

        if action == "template_scroll_click":
            self._edit_template_scroll_click_step(index)
            return

        if action == "template_column_region_click":
            self._edit_template_column_region_click_step(index)
            return

        if action == "wait_region_still":
            self._edit_wait_region_still_step(index)
            return

        if action == "wait_battle_complete":
            self._edit_wait_battle_complete_step(index)
            return

        if action == "swimming_task_loop":
            self._edit_swimming_task_loop_step(index)
            return

        if action == "swipe":
            direction = simpledialog.askstring(
                "修改自然滑动",
                "方向: up / down / left / right",
                initialvalue=str(step.get("direction", "up")),
                parent=self,
            )
            if not direction:
                return
            direction = direction.strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                messagebox.showerror("方向无效", "请输入 up、down、left 或 right。")
                return
            duration_min = simpledialog.askinteger(
                "修改自然滑动", "最短持续毫秒:", initialvalue=int(step.get("duration_min", 650)), minvalue=100, parent=self
            )
            if duration_min is None:
                return
            duration_max = simpledialog.askinteger(
                "修改自然滑动", "最长持续毫秒:", initialvalue=int(step.get("duration_max", 1100)), minvalue=duration_min, parent=self
            )
            if duration_max is None:
                return
            self.steps[index] = {**step, "direction": direction, "duration_min": duration_min, "duration_max": duration_max}
            self._steps_changed(select=index)
            return

        messagebox.showinfo("暂不支持", "这个步骤类型还没有编辑面板，可以先删除后重新添加。")

    def _edit_region_click_step(self, index: int) -> None:
        step = self.steps[index]
        post_delay = step.get("post_delay", [1.5, 2.1])
        pre_delay = step.get("pre_delay", [0.1, 0.35])
        wait_base = float(post_delay[0]) if isinstance(post_delay, list) and len(post_delay) == 2 else 1.5
        wait_extra = round(max(0.0, float(post_delay[1]) - wait_base), 3) if isinstance(post_delay, list) and len(post_delay) == 2 else 0.6
        preview_name = str(step.get("region_preview", ""))
        preview_path = None
        if preview_name:
            preview_path = self.step_previews_dir / preview_name
            if not preview_path.exists():
                preview_path = self.templates_dir / preview_name
        dialog = FormDialog(
            self,
            "修改范围点击",
            [
                ("label", "名称", step.get("label", "范围点击")),
                ("region_text", "点击范围 x1,y1,x2,y2", self._format_pixel_region(step)),
                ("pre_min", "点击前等待最短秒", float(pre_delay[0]) if isinstance(pre_delay, list) and len(pre_delay) == 2 else 0.1),
                ("pre_max", "点击前等待最长秒", float(pre_delay[1]) if isinstance(pre_delay, list) and len(pre_delay) == 2 else 0.35),
                ("wait_base", "点击后基础等待秒", wait_base),
                ("wait_extra", "点击后额外随机秒", wait_extra),
            ],
            extra_actions=[
                ("refresh", "刷新截图"),
                ("set_click_region", "框选设为点击范围"),
                ("apply_selected_template_region", "套用选中模板范围"),
            ],
            preview_path=preview_path,
        )
        if dialog.result is None:
            return
        if dialog.result.get("__action__") == "refresh":
            if self._capture_sync():
                self._edit_region_click_step(index)
            return
        if dialog.result.get("__action__") == "set_click_region":
            if self.screen_image is None and not self._capture_sync():
                return
            if self._selection_relative_region() is not None:
                self._set_step_region_from_selection(index, "region")
                self._edit_region_click_step(index)
                return
            self.pending_step_region_assignment = (index, "region")
            self._show_preview()
            self.selection_text.set("请在左侧截图框选点击范围")
            self._log("等待框选点击范围")
            return
        if dialog.result.get("__action__") == "apply_selected_template_region":
            name = self._selected_template()
            if not name:
                messagebox.showinfo("未选择模板", "请先在模板库选择一个带框选范围的模板。")
                self._edit_region_click_step(index)
                return
            if self._apply_template_region_to_step(index, name):
                self._edit_region_click_step(index)
            return
        try:
            label = dialog.result["label"].strip() or "范围点击"
            pre_min = float(dialog.result["pre_min"])
            pre_max = float(dialog.result["pre_max"])
            wait_base = float(dialog.result["wait_base"])
            wait_extra = float(dialog.result["wait_extra"])
        except ValueError:
            messagebox.showerror("格式错误", "等待时间需要填写数字。")
            return
        if pre_min < 0 or pre_max < pre_min or wait_base < 0 or wait_extra < 0:
            messagebox.showerror("数值无效", "请确认最长等待不小于最短等待。")
            return
        new_step = {
            **step,
            "label": label,
            "use_click_region": True,
            "drift_px": 0,
            "padding_ratio": 0.0,
            "pre_delay": [pre_min, pre_max],
            "post_delay": [wait_base, wait_base + wait_extra],
        }
        new_step.pop("disabled", None)
        self.steps[index] = new_step
        self._steps_changed(select=index)

    def _apply_template_region_to_step(self, index: int, template_name: str) -> bool:
        meta = self._read_template_meta(template_name)
        if not meta:
            messagebox.showinfo("模板没有范围", f"{template_name} 没有框选范围记录。请重新从截图框选保存或覆盖这个模板。")
            return False
        region = meta.get("region")
        source_size = meta.get("source_screen_size")
        pixel_box = meta.get("pixel_box")
        if not (
            isinstance(region, list)
            and len(region) == 4
            and isinstance(source_size, list)
            and len(source_size) == 2
            and isinstance(pixel_box, list)
            and len(pixel_box) == 4
        ):
            messagebox.showinfo("模板范围无效", f"{template_name} 的范围记录不完整，请重新覆盖模板。")
            return False
        screen_w, screen_h = int(source_size[0]), int(source_size[1])
        left, top, right, bottom = (float(value) for value in region)
        center_x = int((left + right) * 0.5 * screen_w)
        center_y = int((top + bottom) * 0.5 * screen_h)
        step = self.steps[index]
        new_step = {
            **step,
            "use_click_region": True,
            "region": [left, top, right, bottom],
            "screen_size": [screen_w, screen_h],
            "pixel_point": [center_x, center_y],
            "point": [center_x / screen_w, center_y / screen_h],
            "drift_px": 0,
            "padding_ratio": 0.0,
            "region_preview": template_name,
        }
        new_step.pop("disabled", None)
        self.steps[index] = new_step
        self._steps_changed(select=index)
        self._log(f"已套用模板 {template_name} 的点击范围: {region}")
        return True

    def _edit_delay_step(self, index: int) -> None:
        step = self.steps[index]
        dialog = FormDialog(
            self,
            "修改随机等待",
            [
                ("min", "最短秒数", step.get("min", 0.5)),
                ("max", "最长秒数", step.get("max", 1.2)),
            ],
        )
        if dialog.result is None:
            return
        try:
            minimum = float(dialog.result["min"])
            maximum = float(dialog.result["max"])
        except ValueError:
            messagebox.showerror("格式错误", "等待时间需要填写数字。")
            return
        if minimum < 0 or maximum < minimum:
            messagebox.showerror("数值无效", "最长秒数不能小于最短秒数。")
            return
        new_step = {"type": "delay", "min": minimum, "max": maximum}
        self.steps[index] = new_step
        self._steps_changed(select=index)

    def _build_region_click_step(
        self,
        existing: dict,
        label: str,
        image_x: int,
        image_y: int,
        image_w: int,
        image_h: int,
        drift: int,
        pre_delay: list[float],
        post_delay: list[float],
    ) -> dict:
        relative_x = image_x / image_w
        relative_y = image_y / image_h
        radius_x = drift / image_w
        radius_y = drift / image_h
        built = {
            **existing,
            "type": existing.get("type", "region_click"),
            "label": label.strip(),
            "pixel_point": [image_x, image_y],
            "screen_size": [image_w, image_h],
            "point": [relative_x, relative_y],
            "drift_px": drift,
            "use_click_region": False,
            "region": [
                max(0.0, relative_x - radius_x),
                max(0.0, relative_y - radius_y),
                min(1.0, relative_x + radius_x),
                min(1.0, relative_y + radius_y),
            ],
            "padding_ratio": 0.0,
            "pre_delay": pre_delay,
            "post_delay": post_delay,
        }
        built.pop("disabled", None)
        return built

    def _format_pixel_region(self, step: dict) -> str:
        region = step.get("region")
        if not isinstance(region, list) or len(region) != 4:
            return "未框选"
        screen_w, screen_h = self._step_screen_size(step)
        left = int(float(region[0]) * screen_w)
        top = int(float(region[1]) * screen_h)
        right = int(float(region[2]) * screen_w)
        bottom = int(float(region[3]) * screen_h)
        if right <= left or bottom <= top:
            return "未框选"
        return f"{left},{top},{right},{bottom}"

    def _format_pixel_region_key(self, step: dict, key: str) -> str:
        region = step.get(key)
        if not isinstance(region, list) or len(region) != 4:
            return "not selected"
        screen_w, screen_h = self._step_screen_size(step)
        left = int(float(region[0]) * screen_w)
        top = int(float(region[1]) * screen_h)
        right = int(float(region[2]) * screen_w)
        bottom = int(float(region[3]) * screen_h)
        if right <= left or bottom <= top:
            return "not selected"
        return f"{left},{top},{right},{bottom}"

    def _step_screen_size(self, step: dict) -> tuple[int, int]:
        screen_size = step.get("screen_size")
        if isinstance(screen_size, list) and len(screen_size) == 2:
            return int(screen_size[0]), int(screen_size[1])
        if self.screen_image is not None:
            image_h, image_w = self.screen_image.shape[:2]
            return image_w, image_h
        size = self.device.screen_size()
        return size.width, size.height

    def _step_pixel_point(self, step: dict, screen_w: int, screen_h: int) -> tuple[int, int]:
        pixel_point = step.get("pixel_point")
        if isinstance(pixel_point, list) and len(pixel_point) == 2:
            return int(pixel_point[0]), int(pixel_point[1])
        point = step.get("point")
        if isinstance(point, list) and len(point) == 2:
            return int(float(point[0]) * screen_w), int(float(point[1]) * screen_h)
        region = step.get("region", [0.0, 0.0, 1.0, 1.0])
        return (
            int((float(region[0]) + float(region[2])) * 0.5 * screen_w),
            int((float(region[1]) + float(region[3])) * 0.5 * screen_h),
        )

    def _edit_template_step(self, index: int) -> None:
        step = self.steps[index]
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先在模板库导入图片，或从左侧手机截图框选保存。")
            return
        initial = step.get("initial_delay", [0.0, 0.0])
        initial_wait = float(initial[0]) if isinstance(initial, list) and initial else 0.0
        dialog = FormDialog(
            self,
            "修改模板步骤",
            [
                ("template", "模板文件名", step.get("template", templates[0])),
                ("threshold", "匹配阈值 0~1", step.get("threshold", 0.88)),
                ("initial_wait", "开始识别前等待秒", initial_wait),
                ("timeout", "最长等待秒数", step.get("timeout", 10.0)),
                ("interval", "轮询间隔秒数", step.get("interval", 0.5)),
                ("search_region", "识别区域，可空", self._format_region(step.get("search_region"))),
                ("fixed_click_region", "固定点击范围，可空", self._format_region(step.get("fixed_click_region"))),
                ("on_timeout", "超时策略", step.get("on_timeout", "error")),
            ],
            extra_actions=[
                ("refresh", "刷新截图"),
                ("set_search_region", "框选设为识别区域"),
                ("set_fixed_click_region", "框选设为点击范围"),
                ("overwrite_template", "用框选覆盖模板"),
            ],
            preview_path=self.templates_dir / str(step.get("template", "")),
        )
        if dialog.result is None:
            return
        if dialog.result.get("__action__") == "refresh":
            if self._capture_sync():
                self._edit_template_step(index)
            return
        if dialog.result.get("__action__") == "overwrite_template":
            self._overwrite_template_from_selection(str(step.get("template", "")))
            return
        if dialog.result.get("__action__") in {"set_search_region", "set_fixed_click_region"}:
            key = "search_region" if dialog.result["__action__"] == "set_search_region" else "fixed_click_region"
            if self._selection_relative_region() is not None:
                self._set_step_region_from_selection(index, key)
                self._edit_template_step(index)
                return
            self.pending_step_region_assignment = (index, key)
            self._show_preview()
            self.selection_text.set("请在左侧截图框选识别区域" if key == "search_region" else "请在左侧截图框选点击范围")
            self._log("等待框选识别区域" if key == "search_region" else "等待框选点击范围")
            return
        name = dialog.result["template"]
        if name and "." not in Path(name).name:
            name = f"{name}.png"
        if not name or name not in templates:
            if name:
                available = "\n".join(templates[:20])
                messagebox.showerror("模板不存在", f"请填写模板库中已有的完整文件名。\n\n可用模板:\n{available}")
            return
        try:
            threshold = float(dialog.result["threshold"])
            initial_wait = float(dialog.result["initial_wait"])
            timeout = float(dialog.result["timeout"])
            interval = float(dialog.result["interval"])
        except ValueError:
            messagebox.showerror("格式错误", "阈值和时间都需要填写数字。")
            return
        on_timeout = dialog.result["on_timeout"].strip()
        on_timeout = on_timeout.strip()
        if threshold < 0 or threshold > 1 or initial_wait < 0 or timeout <= 0 or interval <= 0:
            messagebox.showerror("数值无效", "阈值需要在 0~1 之间，时间需要大于等于 0。")
            return
        if on_timeout not in {"pause", "restart_flow", "error", "continue"}:
            messagebox.showerror("策略无效", "请输入 pause、restart_flow、error 或 continue。")
            return
        new_step = {**step, "template": name, "threshold": threshold, "timeout": timeout, "interval": interval}
        search_region = self._parse_region_text(dialog.result["search_region"]) if dialog.result["search_region"] else None
        fixed_click_region = self._parse_region_text(dialog.result["fixed_click_region"]) if dialog.result["fixed_click_region"] else None
        if dialog.result["search_region"] and search_region is None:
            messagebox.showerror("识别区域无效", "识别区域需要是 left,top,right,bottom，且数值在 0~1。")
            return
        if dialog.result["fixed_click_region"] and fixed_click_region is None:
            messagebox.showerror("点击范围无效", "点击范围需要是 left,top,right,bottom，且数值在 0~1。")
            return
        if search_region is None:
            new_step.pop("search_region", None)
        else:
            new_step["search_region"] = search_region
        if fixed_click_region is None:
            new_step.pop("fixed_click_region", None)
        else:
            new_step["fixed_click_region"] = fixed_click_region
        new_step["on_timeout"] = on_timeout
        if initial_wait > 0:
            new_step["initial_delay"] = [initial_wait, initial_wait + 3.0]
        elif "initial_delay" in new_step:
            del new_step["initial_delay"]
        self.steps[index] = new_step
        self._steps_changed(select=index)

    def _edit_template_scroll_click_step(self, index: int) -> None:
        step = self.steps[index]
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先在模板库导入图片，或从左侧手机截图框选保存。")
            return
        offset = step.get("click_offset", [360, 0])
        dialog = FormDialog(
            self,
            "修改模板滑动重试",
            [
                ("template", "目标模板文件名", step.get("template", templates[0])),
                ("threshold", "匹配阈值 0~1", step.get("threshold", 0.86)),
                ("attempts", "最多尝试次数", step.get("attempts", 2)),
                ("timeout_per_attempt", "每次最多识别秒", step.get("timeout_per_attempt", step.get("timeout", 3))),
                ("interval", "轮询间隔秒", step.get("interval", 0.5)),
                ("click_offset_x", "点击偏移 X 像素", offset[0] if isinstance(offset, list) and len(offset) == 2 else 360),
                ("click_offset_y", "点击偏移 Y 像素", offset[1] if isinstance(offset, list) and len(offset) == 2 else 0),
                ("drift", "点击漂移半径 px", step.get("drift_px", 18)),
                ("pre_swipes", "识别前先滑次数", step.get("pre_swipes", 0)),
                ("swipe_direction", "失败后滑动方向", step.get("swipe_direction", "up")),
                ("on_timeout", "超时策略", step.get("on_timeout", "pause")),
            ],
            extra_actions=[("refresh", "刷新截图"), ("overwrite_template", "用框选覆盖模板")],
            preview_path=self.templates_dir / str(step.get("template", "")),
        )
        if dialog.result is None:
            return
        if dialog.result.get("__action__") == "refresh":
            if self._capture_sync():
                self._edit_template_scroll_click_step(index)
            return
        if dialog.result.get("__action__") == "overwrite_template":
            self._overwrite_template_from_selection(str(step.get("template", "")))
            return
        new_step = self._parse_template_scroll_click_form(dialog.result, templates, existing=step)
        if new_step is None:
            return
        self.steps[index] = new_step
        self._steps_changed(select=index)

    def _edit_template_column_region_click_step(self, index: int) -> None:
        step = self.steps[index]
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先在模板库导入图片，或从左侧手机截图框选保存。")
            return
        regions = step.get("column_click_regions")
        if not isinstance(regions, dict):
            regions = {}
        dialog = FormDialog(
            self,
            "修改按列范围点击",
            [
                ("template", "目标卡片模板文件名", step.get("template", templates[0])),
                ("threshold", "匹配阈值 0~1", step.get("threshold", 0.84)),
                ("attempts", "最多尝试次数", step.get("attempts", 2)),
                ("timeout_per_attempt", "每次最多识别秒", step.get("timeout_per_attempt", step.get("timeout", 3))),
                ("interval", "轮询间隔秒", step.get("interval", 0.5)),
                ("pre_swipes", "识别前先滑次数", step.get("pre_swipes", 0)),
                ("lock_column", "第一次识别后锁定列 true/false", str(step.get("lock_column", False)).lower()),
                ("left_region", "左列参加范围", self._format_region(regions.get("left"))),
                ("right_region", "右列参加范围", self._format_region(regions.get("right"))),
                ("on_timeout", "超时策略", step.get("on_timeout", "pause")),
            ],
            extra_actions=[
                ("refresh", "刷新截图"),
                ("set_left_region", "框选设为左列范围"),
                ("set_right_region", "框选设为右列范围"),
            ],
            preview_path=self.templates_dir / str(step.get("template", "")),
        )
        if dialog.result is None:
            return
        action = dialog.result.get("__action__")
        if action == "refresh":
            if self._capture_sync():
                self._edit_template_column_region_click_step(index)
            return
        if action in {"set_left_region", "set_right_region"}:
            side = "left" if action == "set_left_region" else "right"
            if self._selection_relative_region() is not None:
                self._set_column_click_region_from_selection(index, side)
                self._edit_template_column_region_click_step(index)
                return
            self.pending_column_region_assignment = (index, side)
            self._show_preview()
            self.selection_text.set("Select left participate button area" if side == "left" else "Select right participate button area")
            self._log("Waiting for left participate button area selection" if side == "left" else "Waiting for right participate button area selection")
            return


        new_step = self._parse_template_column_region_click_form(dialog.result, templates, existing=step)
        if new_step is None:
            return
        self.steps[index] = new_step
        self._steps_changed(select=index)

    def _parse_template_column_region_click_form(self, values: dict[str, str], templates: list[str], existing: dict | None = None) -> dict | None:
        existing = existing or {}
        name = values["template"]
        if name and "." not in Path(name).name:
            name = f"{name}.png"
        if not name or name not in templates:
            if name:
                available = "\n".join(templates[:20])
                messagebox.showerror("模板不存在", f"请填写模板库中已有的完整文件名。\n\n可用模板:\n{available}")
            return None
        left_region = self._parse_region_text(values["left_region"]) if values["left_region"] else None
        right_region = self._parse_region_text(values["right_region"]) if values["right_region"] else None
        if values["left_region"] and left_region is None:
            messagebox.showerror("左列范围无效", "左列范围需要是 left,top,right,bottom，且数值在 0~1。")
            return None
        if values["right_region"] and right_region is None:
            messagebox.showerror("右列范围无效", "右列范围需要是 left,top,right,bottom，且数值在 0~1。")
            return None
        try:
            threshold = float(values["threshold"])
            attempts = int(float(values["attempts"]))
            timeout_per_attempt = float(values["timeout_per_attempt"])
            interval = float(values["interval"])
            pre_swipes = int(float(values["pre_swipes"]))
        except ValueError:
            messagebox.showerror("格式错误", "阈值、次数、时间和先滑次数都需要填写数字。")
            return None
        lock_text = values["lock_column"].strip().lower()
        if lock_text not in {"true", "false", "1", "0", "yes", "no"}:
            messagebox.showerror("锁定列无效", "请填写 true 或 false。")
            return None
        on_timeout = values["on_timeout"].strip()
        if threshold < 0 or threshold > 1 or attempts <= 0 or timeout_per_attempt <= 0 or interval <= 0 or pre_swipes < 0:
            messagebox.showerror("数值无效", "请确认阈值在 0~1，次数和时间大于 0，先滑次数不小于 0。")
            return None
        if on_timeout not in {"pause", "restart_flow", "error", "continue"}:
            messagebox.showerror("策略无效", "请输入 pause、restart_flow、error 或 continue。")
            return None
        regions: dict[str, list[float]] = {}
        if left_region:
            regions["left"] = left_region
        if right_region:
            regions["right"] = right_region
        return {
            **existing,
            "type": "template_column_region_click",
            "template": name,
            "threshold": threshold,
            "attempts": attempts,
            "timeout_per_attempt": timeout_per_attempt,
            "interval": interval,
            "pre_swipes": pre_swipes,
            "lock_column": lock_text in {"true", "1", "yes"},
            "lock_key": str(existing.get("lock_key") or name),
            "column_click_regions": regions,
            "padding_ratio": float(existing.get("padding_ratio", 0.18)),
            "pre_delay": existing.get("pre_delay", [0.12, 0.35]),
            "post_delay": existing.get("post_delay", [1.5, 2.3]),
            "swipe_direction": existing.get("swipe_direction", "up"),
            "swipe_duration_min": int(existing.get("swipe_duration_min", 550)),
            "swipe_duration_max": int(existing.get("swipe_duration_max", 850)),
            "after_swipe_delay": existing.get("after_swipe_delay", [0.8, 1.3]),
            "on_timeout": on_timeout,
            "screenshot_on_timeout": True,
            "debug_screenshot_after": True,
        }

    def _parse_template_scroll_click_form(self, values: dict[str, str], templates: list[str], existing: dict | None = None) -> dict | None:
        existing = existing or {}
        name = values["template"]
        if name and "." not in Path(name).name:
            name = f"{name}.png"
        if not name or name not in templates:
            if name:
                available = "\n".join(templates[:20])
                messagebox.showerror("模板不存在", f"请填写模板库中已有的完整文件名。\n\n可用模板:\n{available}")
            return None
        try:
            threshold = float(values["threshold"])
            attempts = int(float(values["attempts"]))
            timeout_per_attempt = float(values["timeout_per_attempt"])
            interval = float(values["interval"])
            offset_x = int(float(values["click_offset_x"]))
            offset_y = int(float(values["click_offset_y"]))
            drift = int(float(values["drift"]))
            pre_swipes = int(float(values["pre_swipes"]))
        except ValueError:
            messagebox.showerror("格式错误", "阈值、次数、时间、偏移、漂移和先滑次数都需要填写数字。")
            return None
        swipe_direction = values["swipe_direction"].strip().lower()
        on_timeout = values["on_timeout"].strip()
        if threshold < 0 or threshold > 1 or attempts <= 0 or timeout_per_attempt <= 0 or interval <= 0 or drift < 0 or pre_swipes < 0:
            messagebox.showerror("数值无效", "请确认阈值在 0~1，次数和时间大于 0，漂移和先滑次数不小于 0。")
            return None
        if swipe_direction not in {"up", "down", "left", "right"}:
            messagebox.showerror("滑动方向无效", "请输入 up、down、left 或 right。")
            return None
        if on_timeout not in {"pause", "restart_flow", "error", "continue"}:
            messagebox.showerror("策略无效", "请输入 pause、restart_flow、error 或 continue。")
            return None
        return {
            **existing,
            "type": "template_scroll_click",
            "template": name,
            "threshold": threshold,
            "attempts": attempts,
            "timeout_per_attempt": timeout_per_attempt,
            "interval": interval,
            "click_offset": [offset_x, offset_y],
            "drift_px": drift,
            "pre_swipes": pre_swipes,
            "swipe_direction": swipe_direction,
            "swipe_duration_min": int(existing.get("swipe_duration_min", 550)),
            "swipe_duration_max": int(existing.get("swipe_duration_max", 850)),
            "after_swipe_delay": existing.get("after_swipe_delay", [0.7, 1.2]),
            "pre_delay": existing.get("pre_delay", [0.12, 0.35]),
            "post_delay": existing.get("post_delay", [1.5, 2.3]),
            "on_timeout": on_timeout,
            "screenshot_on_timeout": True,
        }

    def _edit_wait_region_still_step(self, index: int) -> None:
        step = self.steps[index]
        reselect = self.screen_image is not None and messagebox.askyesno(
            "修改区域稳定步骤",
            "是否用左侧当前截图里的框选区域替换观察区域？\n\n选择“否”只修改名称和时间参数。",
            parent=self,
        )
        region = step.get("region", [0.0, 0.0, 1.0, 1.0])
        if reselect:
            if not self.selection_box:
                messagebox.showinfo("提示", "请先在左侧截图里框选一个区域。")
                return
            image_h, image_w = self.screen_image.shape[:2]
            left, top, right, bottom = self.selection_box
            region = [left / image_w, top / image_h, right / image_w, bottom / image_h]
        label = simpledialog.askstring(
            "修改区域稳定步骤",
            "步骤名称:",
            initialvalue=str(step.get("label", "等待区域稳定")),
            parent=self,
        )
        if not label:
            return
        timeout = simpledialog.askfloat(
            "修改区域稳定步骤",
            "最多等待多少秒:",
            initialvalue=float(step.get("timeout", 480.0)),
            minvalue=1.0,
            maxvalue=3600.0,
            parent=self,
        )
        if timeout is None:
            return
        interval = simpledialog.askfloat(
            "修改区域稳定步骤",
            "每隔多少秒截图比较一次:",
            initialvalue=float(step.get("interval", 5.0)),
            minvalue=1.0,
            maxvalue=60.0,
            parent=self,
        )
        if interval is None:
            return
        stable_count = simpledialog.askinteger(
            "修改区域稳定步骤",
            "连续稳定几次才算结束:",
            initialvalue=int(step.get("stable_count", 3)),
            minvalue=1,
            maxvalue=20,
            parent=self,
        )
        if stable_count is None:
            return
        on_timeout = simpledialog.askstring(
            "修改区域稳定步骤",
            "超时策略: pause / restart_flow / error / continue",
            initialvalue=str(step.get("on_timeout", "pause")),
            parent=self,
        )
        if not on_timeout:
            return
        on_timeout = on_timeout.strip()
        if on_timeout not in {"pause", "restart_flow", "error", "continue"}:
            messagebox.showerror("策略无效", "请输入 pause、restart_flow、error 或 continue。")
            return
        self.steps[index] = {
            **step,
            "label": label.strip(),
            "region": region,
            "timeout": timeout,
            "interval": interval,
            "stable_count": stable_count,
            "on_timeout": on_timeout,
        }
        self._steps_changed(select=index)

    def _edit_wait_battle_complete_step(self, index: int) -> None:
        step = self.steps[index]
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("还没有模板", "请先保存顶部“回合”文字模板，例如：战斗回合标志。")
            return
        dialog = FormDialog(
            self,
            "修改等待战斗结束",
            [
                ("template", "战斗标志模板文件名", step.get("template", templates[0])),
                ("threshold", "匹配阈值 0~1", step.get("threshold", 0.86)),
                ("enter_timeout", "寻路/等待进入战斗最多秒", step.get("enter_timeout", 240.0)),
                ("battle_timeout", "进入战斗后最多等待秒", step.get("battle_timeout", 180.0)),
                ("interval", "每隔多少秒识别一次", step.get("interval", 5.0)),
                ("confirm_gone", "连续几次未识别到才算结束", step.get("confirm_gone", 3)),
                ("on_timeout", "超时策略", step.get("on_timeout", "pause")),
            ],
            extra_actions=[("refresh", "刷新截图"), ("overwrite_template", "用框选覆盖模板")],
        )
        if dialog.result is None:
            return
        if dialog.result.get("__action__") == "refresh":
            if self._capture_sync():
                self._edit_wait_battle_complete_step(index)
            return
        if dialog.result.get("__action__") == "overwrite_template":
            self._overwrite_template_from_selection(str(step.get("template", "")))
            return
        name = dialog.result["template"]
        if name and "." not in Path(name).name:
            name = f"{name}.png"
        if not name or name not in templates:
            if name:
                available = "\n".join(templates[:20])
                messagebox.showerror("模板不存在", f"请填写模板库中已有的完整文件名。\n\n可用模板:\n{available}")
            return
        try:
            threshold = float(dialog.result["threshold"])
            enter_timeout = float(dialog.result["enter_timeout"])
            battle_timeout = float(dialog.result["battle_timeout"])
            interval = float(dialog.result["interval"])
            confirm_gone = int(float(dialog.result["confirm_gone"]))
        except ValueError:
            messagebox.showerror("格式错误", "阈值、时间和确认次数都需要填写数字。")
            return
        on_timeout = dialog.result["on_timeout"].strip()
        if threshold < 0 or threshold > 1 or enter_timeout <= 0 or battle_timeout <= 0 or interval <= 0 or confirm_gone <= 0:
            messagebox.showerror("数值无效", "阈值需要在 0~1 之间，时间和确认次数需要大于 0。")
            return
        if on_timeout not in {"pause", "restart_flow", "error", "continue"}:
            messagebox.showerror("策略无效", "请输入 pause、restart_flow、error 或 continue。")
            return
        new_step = {
            **step,
            "template": name,
            "threshold": threshold,
            "enter_timeout": enter_timeout,
            "battle_timeout": battle_timeout,
            "interval": interval,
            "confirm_gone": confirm_gone,
            "search_region": step.get("search_region", [0.32, 0.0, 0.68, 0.18]),
            "on_timeout": on_timeout,
            "screenshot_on_timeout": True,
        }
        new_step.pop("disabled", None)
        self.steps[index] = new_step
        self._steps_changed(select=index)

    def _edit_swimming_task_loop_step(self, index: int) -> None:
        step = self.steps[index]
        templates = self._template_names()
        if not templates:
            messagebox.showinfo("No templates", "Please save the battle-round marker template first.")
            return
        dialog = FormDialog(
            self,
            "Edit Swimming Loop",
            [
                ("template", "Battle marker template", step.get("template", templates[0])),
                ("threshold", "Match threshold 0~1", step.get("threshold", 0.8)),
                ("task_region_text", "Task click region x1,y1,x2,y2", self._format_pixel_region_key(step, "task_region")),
                ("idle_min", "Idle click min seconds", (step.get("idle_delay", [2.0, 4.0]) or [2.0, 4.0])[0]),
                ("idle_max", "Idle click max seconds", (step.get("idle_delay", [2.0, 4.0]) or [2.0, 4.0])[1]),
                ("check_interval", "Battle check interval seconds", step.get("check_interval", 0.65)),
                ("battle_timeout", "Battle max wait seconds", step.get("battle_timeout", 300.0)),
                ("max_duration", "Loop max duration seconds", step.get("max_duration", 1800.0)),
                ("max_task_clicks", "Max task clicks", step.get("max_task_clicks", 300)),
                ("on_timeout", "Timeout policy", step.get("on_timeout", "pause")),
            ],
            extra_actions=[
                ("refresh", "Refresh Screenshot"),
                ("set_task_region", "Select Task Region"),
            ],
        )
        if dialog.result is None:
            return
        if dialog.result.get("__action__") == "refresh":
            if self._capture_sync():
                self._edit_swimming_task_loop_step(index)
            return
        if dialog.result.get("__action__") == "set_task_region":
            if self.screen_image is None and not self._capture_sync():
                return
            if self._selection_relative_region() is not None:
                self._set_step_region_from_selection(index, "task_region")
                self._edit_swimming_task_loop_step(index)
                return
            self.pending_step_region_assignment = (index, "task_region")
            self._show_preview()
            self.selection_text.set("Select the first task row on the right task panel")
            self._log("waiting for swimming task region selection")
            return
        name = dialog.result["template"]
        if name and "." not in Path(name).name:
            name = f"{name}.png"
        if not name or name not in templates:
            messagebox.showerror("Template not found", "Please use an existing template file name.")
            return
        try:
            threshold = float(dialog.result["threshold"])
            idle_min = float(dialog.result["idle_min"])
            idle_max = float(dialog.result["idle_max"])
            check_interval = float(dialog.result["check_interval"])
            battle_timeout = float(dialog.result["battle_timeout"])
            max_duration = float(dialog.result["max_duration"])
            max_task_clicks = int(float(dialog.result["max_task_clicks"]))
        except ValueError:
            messagebox.showerror("Invalid number", "Timing and threshold fields must be numbers.")
            return
        if threshold < 0 or threshold > 1 or idle_min < 0 or idle_max < idle_min or check_interval <= 0 or battle_timeout <= 0 or max_duration <= 0 or max_task_clicks <= 0:
            messagebox.showerror("Invalid value", "Please check thresholds, timings and limits.")
            return
        on_timeout = dialog.result["on_timeout"].strip() or "pause"
        if on_timeout not in {"pause", "restart_flow", "error", "continue"}:
            messagebox.showerror("Invalid policy", "Use pause, restart_flow, error or continue.")
            return
        new_step = {
            **step,
            "template": name,
            "threshold": threshold,
            "idle_delay": [idle_min, idle_max],
            "check_interval": check_interval,
            "battle_timeout": battle_timeout,
            "max_duration": max_duration,
            "max_task_clicks": max_task_clicks,
            "on_timeout": on_timeout,
            "screenshot_on_timeout": True,
        }
        new_step.pop("disabled", None)
        self.steps[index] = new_step
        self._steps_changed(select=index)

    def _run_flow(self, start_index: int = 0) -> None:
        if not self.flow_selected_for_run or self.flow_path is None:
            messagebox.showinfo("请选择流程", "请先在流程下拉框里选择一个流程，再点击 Run All。")
            self._log("未选择流程，已阻止运行。")
            return
        if not self.steps:
            messagebox.showinfo("????", "?????????????")
            return
        if self.worker and self.worker.is_alive():
            return
        start_index = max(0, min(start_index, len(self.steps) - 1))
        self._save_flow()
        self._log(f"开始运行流程: {self.flow_path.stem}")
        self.stop_event = threading.Event()
        frame_source = None
        if self.flow_frame_source == "video":
            self._log("using experimental video frame source for this flow")
            frame_source = ScrcpyVideoFrameSource(self.device)
        runner = WorkflowRunner(self.device, self.templates_dir, self._thread_log, self.stop_event, frame_source=frame_source)
        steps = [dict(step) for step in self.steps]
        repeat = max(0, self.repeat_var.get())
        self.run_button.configure(state="disabled")
        self.run_from_selected_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        def task() -> None:
            try:
                runner.run(steps, repeat, start_index=start_index)
            except Exception as exc:
                self.events.put(("error", f"????: {exc}"))
            finally:
                self.events.put(("finished", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _run_flow_from_selected(self) -> None:
        selection = self.step_list.curselection()
        if not selection:
            messagebox.showinfo("?????", "???????????????????")
            return
        self._run_flow(start_index=selection[0])

    def _stop_flow(self) -> None:
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self._log("正在停止流程...")

    def _start_scrcpy(self) -> None:
        try:
            subprocess.Popen(["scrcpy"], creationflags=subprocess.CREATE_NO_WINDOW)
            self._log("已启动 scrcpy")
        except FileNotFoundError:
            messagebox.showerror("scrcpy not found", "scrcpy.exe was not found, recording cannot start.")
        except Exception as exc:
            messagebox.showerror("启动失败", str(exc))


    def _toggle_recording(self) -> None:
        if self.recording_worker and self.recording_worker.is_alive():
            self._stop_recording()
        else:
            self._start_segment_recording()

    def _start_segment_recording(self) -> None:
        scrcpy = self._scrcpy_path()
        if scrcpy is None:
            messagebox.showerror("scrcpy not found", "scrcpy.exe was not found, recording cannot start.")
            return
        if self.recording_worker and self.recording_worker.is_alive():
            return
        self.recordings_dir.mkdir(exist_ok=True)
        self.recording_event.clear()
        self.record_button.configure(text="Stop Rec", state="normal")
        self._log(f"recording started: {self.recordings_dir}")
        self.recording_worker = threading.Thread(target=self._recording_loop, args=(scrcpy,), daemon=True)
        self.recording_worker.start()

    def _stop_recording(self) -> None:
        self.recording_event.set()
        process = self.recording_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        self.record_button.configure(state="disabled")
        self._log("stopping recording...")

    def _recording_loop(self, scrcpy: str) -> None:
        segment = 1
        try:
            while not self.recording_event.is_set():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = self.recordings_dir / f"screen_{stamp}_{segment:03d}.mp4"
                stdout = output.with_suffix(".out.log")
                stderr = output.with_suffix(".err.log")
                args = [
                    "--no-window",
                    "--record",
                    str(output),
                    "--record-format",
                    "mp4",
                    "--video-bit-rate",
                    "4M",
                    "--max-fps",
                    "15",
                    "--time-limit",
                    str(int(self.recording_segment_seconds)),
                ]
                serial = self._current_adb_serial()
                if serial:
                    args = ["--serial", serial, *args]
                self.events.put(("log", f"recording segment {segment}: {output}"))
                with stdout.open("wb") as out_file, stderr.open("wb") as err_file:
                    process = subprocess.Popen(
                        [scrcpy, *args],
                        cwd=str(ROOT),
                        stdout=out_file,
                        stderr=err_file,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    )
                    self.recording_process = process
                    while not self.recording_event.is_set():
                        if process.poll() is not None:
                            break
                        time.sleep(0.5)
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=6)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                self.recording_process = None
                segment += 1
        except Exception as exc:
            self.events.put(("log", f"recording failed: {exc}"))
        finally:
            self.recording_process = None
            self.recording_event.set()
            self.events.put(("recording_stopped", "recording stopped"))

    def _scrcpy_path(self) -> str | None:
        bundled = Path("D:/soft/scrcpy/scrcpy.exe")
        if bundled.exists():
            return str(bundled)
        found = shutil.which("scrcpy")
        return found

    def _current_adb_serial(self) -> str | None:
        try:
            result = self.device.run("devices", "-l", check=False)
            devices = []
            for line in result.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            return devices[0] if len(devices) == 1 else None
        except Exception:
            return None

    def _refresh_templates(self, select: str | None = None) -> None:
        self.template_list.delete(0, "end")
        names = self._template_names()
        for name in names:
            self.template_list.insert("end", name)
        if select in names:
            index = names.index(select)
            self.template_list.selection_set(index)
            self.template_list.see(index)
            self._show_template_preview(select)

    def _selected_template(self) -> str | None:
        selection = self.template_list.curselection()
        return self.template_list.get(selection[0]) if selection else None

    def _show_selected_template_preview(self, _event: tk.Event | None = None) -> None:
        if _event is not None and getattr(_event, "type", None) is not None:
            self.after(10, self._show_selected_template_preview)
            return
        name = self._selected_template()
        if name:
            self._show_template_preview(name)

    def _show_template_preview(self, name: str) -> None:
        path = self.templates_dir / name
        try:
            data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        except OSError:
            data = np.array([], dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            self.template_preview_photo = None
            self.template_preview_label.configure(text=f"无法读取模板: {name}", image="")
            return
        height, width = image.shape[:2]
        max_w, max_h = 360, 140
        scale = min(max_w / max(1, width), max_h / max(1, height), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            self.template_preview_photo = None
            self.template_preview_label.configure(text=f"无法预览模板: {name}", image="")
            return
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.template_preview_photo = tk.PhotoImage(data=data)
        meta = self._read_template_meta(name)
        meta_text = ""
        if meta:
            pixel_box = meta.get("pixel_box", [])
            source_size = meta.get("source_screen_size", [])
            if isinstance(pixel_box, list) and len(pixel_box) == 4 and isinstance(source_size, list) and len(source_size) == 2:
                meta_text = f"\n框选范围: {pixel_box[0]},{pixel_box[1]},{pixel_box[2]},{pixel_box[3]} / {source_size[0]}x{source_size[1]}"
        self.template_preview_label.configure(
            text=f"{name} · {width} × {height}{meta_text}",
            image=self.template_preview_photo,
            compound="top",
        )

    def _template_names(self) -> list[str]:
        suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        return sorted(path.name for path in self.templates_dir.iterdir() if path.suffix.lower() in suffixes)

    def _refresh_step_list(self, select: int | None = None) -> None:
        self.step_list.delete(0, "end")
        for index, step in enumerate(self.steps, start=1):
            self.step_list.insert("end", f"{index}. {WorkflowRunner.describe_step(step)}")
        if select is not None and self.steps:
            self.step_list.selection_set(select)

    def _steps_changed(self, select: int | None = None) -> None:
        self._refresh_step_list(select=select)
        self._save_flow()

    def _load_flow(self) -> None:
        self._refresh_flow_selector()
        if self.flow_path is None:
            self.steps = []
            self.flow_frame_source = "adb"
            self._refresh_step_list()
            return
        if not self.flow_path.exists():
            self.steps = []
            self.flow_frame_source = "adb"
            self._refresh_step_list()
            return
        try:
            data = json.loads(self.flow_path.read_text(encoding="utf-8-sig"))
            self.steps = data.get("steps", [])
            self.flow_frame_source = str(data.get("frame_source", "adb")).strip().lower() or "adb"
            self.repeat_var.set(max(0, int(data.get("repeat", 0))))
            self._refresh_step_list()
        except Exception as exc:
            self._log(f"读取流程配置失败: {exc}")

    def _save_flow(self) -> None:
        if self.flow_path is None:
            self._log("未选择流程，跳过保存。")
            return
        data = {"repeat": max(0, self.repeat_var.get()), "steps": self.steps}
        if self.flow_frame_source != "adb":
            data["frame_source"] = self.flow_frame_source
        self.flow_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _refresh_flow_selector(self) -> None:
        names = sorted(path.stem for path in self.flows_dir.glob("*.json"))
        current = self.flow_path.stem if self.flow_path is not None else ""
        if current and current not in names:
            names.append(current)
        self.flow_selector.configure(values=names)
        self.flow_name_var.set(current)

    def _select_flow(self, _event: tk.Event | None = None) -> None:
        name = self.flow_name_var.get().strip()
        if not name:
            return
        self.flow_path = self.flows_dir / f"{name}.json"
        self.flow_selected_for_run = True
        self._load_flow()
        self.run_button.configure(state="normal")
        self.run_from_selected_button.configure(state="normal")
        self._log(f"已切换流程: {name}")

    def _new_flow(self) -> None:
        name = simpledialog.askstring("新建流程", "流程名称:", parent=self)
        if not name:
            return
        safe_name = "".join(char for char in name.strip() if char not in '\\/:*?"<>|').strip()
        if not safe_name:
            messagebox.showerror("名称无效", "请输入有效的流程名称。")
            return
        path = self.flows_dir / f"{safe_name}.json"
        if path.exists() and not messagebox.askyesno("流程已存在", "同名流程已存在，是否打开？"):
            return
        self.flow_path = path
        self.flow_selected_for_run = True
        self.steps = []
        self.repeat_var.set(0)
        self._steps_changed()
        self._refresh_flow_selector()
        self.run_button.configure(state="normal")
        self.run_from_selected_button.configure(state="normal")
        self._log(f"已新建流程: {safe_name}")

    def _thread_log(self, message: str) -> None:
        self.events.put(("log", message))

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        try:
            self.log_path.parent.mkdir(exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line)
        except OSError:
            pass
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.device_status.set(str(value))
                elif kind == "log":
                    self._log(str(value))
                elif kind == "capture":
                    self._capture_async()
                elif kind == "image":
                    self.screen_image = value  # type: ignore[assignment]
                    self.selection_text.set("拖动鼠标框选要识别的按钮或图标")
                    self._render_preview()
                elif kind == "error":
                    self._log(str(value))
                    messagebox.showerror("错误", str(value))
                elif kind == "info":
                    messagebox.showinfo("识别测试", str(value))
                elif kind == "finished":
                    run_state = "normal" if self.flow_selected_for_run and self.flow_path is not None else "disabled"
                    self.run_button.configure(state=run_state)
                    self.run_from_selected_button.configure(state=run_state)
                    self.stop_button.configure(state="disabled")
                elif kind == "recording_stopped":
                    self.record_button.configure(text="Start Rec", state="normal")
                    self._log(str(value))
        except queue.Empty:
            pass
        self.after(100, self._drain_events)


if __name__ == "__main__":
    ToolboxApp().mainloop()
