"""
BENIROBO - 3D Vision Robot Automation UI
실행: python3 vision_ui.py
의존: pip install PyQt6 open3d
"""

import sys, time, os, math
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QLabel, QFrame, QPushButton, QTextEdit,
    QComboBox, QSizePolicy, QScrollArea, QStackedWidget,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QFileDialog, QSlider, QSplitter,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QPainterPath, QFont

# ── 팔레트 ─────────────────────────────────────────────────────────
BRAND     = "#00AAFF"
BG_APP    = "#F2F2F2"
BG_WHITE  = "#FFFFFF"
BG_HEADER = "#EBEBEB"
BG_SIDE   = "#F5F5F5"
BG_CANVAS = "#FAFAFA"
BG_LOG    = "#F7F7F7"
BG_TAB    = "#EEEEEE"
BG_HOVER  = "#E8F5FF"
BG_BTN    = "#F0F0F0"
TEXT_PRI  = "#1A1A1A"
TEXT_SEC  = "#555555"
TEXT_DIM  = "#999999"
TEXT_LOG  = "#333333"
BORDER    = "#888888"
BORDER_LT = "#CCCCCC"
SUCCESS   = "#28A745"
DANGER    = "#DC3545"
WARN      = "#FFC107"

# ── 공통 구분선 ────────────────────────────────────────────────────
def hline():
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1); f.setStyleSheet(f"background:{BORDER}; border:none;")
    return f

def vline():
    f = QFrame(); f.setFrameShape(QFrame.Shape.VLine)
    f.setFixedWidth(1); f.setStyleSheet(f"background:{BORDER}; border:none;")
    return f

def _tab_btn_style(active: bool) -> str:
    if active:
        return f"""QPushButton {{
            background:{BG_WHITE}; color:{BRAND};
            border:1px solid {BORDER}; border-bottom:none;
            border-radius:4px 4px 0 0;
            font-size:11px; font-weight:700; padding:0 12px;
        }}"""
    return f"""QPushButton {{
        background:{BG_TAB}; color:{TEXT_SEC};
        border:1px solid {BORDER_LT}; border-bottom:none;
        border-radius:4px 4px 0 0; font-size:11px; padding:0 12px;
    }}
    QPushButton:hover {{ background:{BG_HOVER}; color:{BRAND}; }}"""

def _field_style():
    return f"""
        QSpinBox, QDoubleSpinBox, QComboBox {{
            background:{BG_WHITE}; color:{TEXT_PRI};
            border:1px solid {BORDER_LT}; border-radius:4px;
            padding:3px 6px; font-size:11px; min-height:24px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color:{BRAND}; }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width:18px; border:none; background:{BG_BTN};
        }}
        QComboBox::drop-down {{ border:none; width:20px; }}
        QComboBox QAbstractItemView {{
            background:{BG_WHITE}; color:{TEXT_PRI};
            selection-background-color:{BG_HOVER};
        }}
    """

def _group_style():
    return f"""
        QGroupBox {{
            color:{TEXT_SEC}; font-size:11px; font-weight:700;
            border:1px solid {BORDER_LT}; border-radius:6px;
            margin-top:10px; padding-top:6px;
        }}
        QGroupBox::title {{
            subcontrol-origin:margin; subcontrol-position:top left;
            left:10px; padding:0 4px;
            color:{BRAND}; font-size:11px;
        }}
    """

def _apply_btn(text, color=BRAND):
    b = QPushButton(text)
    b.setFixedHeight(28)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{color}18; color:{color};
            border:1px solid {color}80; border-radius:5px;
            font-size:11px; font-weight:600; padding:0 14px;
        }}
        QPushButton:hover {{ background:{color}33; }}
        QPushButton:pressed {{ background:{color}; color:#fff; }}
    """)
    return b


# ══════════════════════════════════════════════════════════════════
# UIBridge 스텁
# ══════════════════════════════════════════════════════════════════
class UIBridge:
    def __init__(self, win):
        self.win = win; self.runner = None

    def connect_signals(self): pass

    def start_pipeline(self):
        self.win.log.push("UIBridge: PipelineRunner 미연결 — 스텁 실행", "WARN")

    def stop_pipeline(self):
        self.win.log.push("UIBridge: 파이프라인 중지 요청", "INFO")

    def load_config(self) -> dict:
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        if not os.path.exists(cfg_path):
            self.win.log.push(f"config.yaml 없음: {cfg_path}", "WARN"); return {}
        try:
            import yaml
            with open(cfg_path) as f: return yaml.safe_load(f) or {}
        except Exception as e:
            self.win.log.push(f"config 로드 오류: {e}", "ERR"); return {}

    def save_config(self, config: dict):
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        try:
            import yaml
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, "w") as f: yaml.dump(config, f, allow_unicode=True)
            self.win.log.push("설정 저장 완료.", "OK")
        except Exception as e:
            self.win.log.push(f"설정 저장 오류: {e}", "ERR")


# ══════════════════════════════════════════════════════════════════
# 타이틀 바
# ══════════════════════════════════════════════════════════════════
class TitleBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(48)
        self.setStyleSheet(f"background:{BG_WHITE};")
        h = QHBoxLayout(self)
        h.setContentsMargins(16, 0, 16, 0); h.setSpacing(0)

        logo = QLabel("BENIROBO")
        logo.setStyleSheet(f"color:{BRAND}; font-size:22px; font-weight:800; letter-spacing:1px;")
        h.addWidget(logo); h.addStretch()

        status_w = QWidget(); status_w.setStyleSheet(f"background:{BG_WHITE};")
        sv = QVBoxLayout(status_w); sv.setContentsMargins(10,4,10,4); sv.setSpacing(2)
        sv.addWidget(self._lbl("현재 상태 표기 구역", f"color:{TEXT_SEC}; font-size:10px; font-weight:600;"))
        sv.addWidget(self._lbl("(적용카메라, 아이피, 작동상태, 경고등)", f"color:{TEXT_DIM}; font-size:9px;"))

        sh = QHBoxLayout(); sh.setContentsMargins(0,0,0,0); sh.setSpacing(8)
        self._ind = {}
        for key, color in [("CAM",BRAND),("IP",SUCCESS),("RUNNING",TEXT_DIM),("⚠",WARN)]:
            lbl = QLabel(f"● {key}")
            lbl.setStyleSheet(f"color:{color}; font-size:10px; font-weight:600;")
            self._ind[key] = lbl; sh.addWidget(lbl)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG_WHITE};"); inner.setLayout(sh)
        sv.addWidget(inner); h.addWidget(status_w)

    @staticmethod
    def _lbl(text, style):
        l = QLabel(text); l.setStyleSheet(style); return l

    def set_indicator(self, key: str, active: bool):
        colors = {"CAM":(BRAND,TEXT_DIM),"IP":(SUCCESS,TEXT_DIM),
                  "RUNNING":(DANGER,TEXT_DIM),"⚠":(WARN,TEXT_DIM)}
        on_col, off_col = colors.get(key, (SUCCESS, TEXT_DIM))
        if key in self._ind:
            col = on_col if active else off_col
            self._ind[key].setStyleSheet(f"color:{col}; font-size:10px; font-weight:600;")


# ══════════════════════════════════════════════════════════════════
# 탭 메타데이터
# ══════════════════════════════════════════════════════════════════
TAB_NAMES  = ["Run","Detection","Camera","Calibration","Pick Point","I/O","Setting"]
SIDE_MENUS = [
    [("파이프라인 시작",),("Step 실행",),("일시정지",),("중지",)],
    [("RTMDet 설정",),("신뢰도 임계값",),("클래스 선택",)],
    [("해상도",),("FPS",),("노출값",),("드라이버 교체",)],
    [("Hand-Eye 방식",),("내부 파라미터",),("외부 파라미터",),("결과 저장/로드",)],
    [("그립 포인트 지정",),("오프셋 설정",),("후보 랭킹",),("JSON 저장/로드",)],
    [("포트 설정",),("TCP 주소",),("메세지 포맷",)],
    [("언어",),("저장 경로",),("로그 레벨",)],
]


# ══════════════════════════════════════════════════════════════════
# 서브 툴바
# ══════════════════════════════════════════════════════════════════
class SubToolBar(QWidget):
    tab_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(38)
        self.setStyleSheet(f"background:{BG_HEADER};")
        h = QHBoxLayout(self); h.setContentsMargins(0,0,0,0); h.setSpacing(0)

        left_w = QWidget(); left_w.setFixedWidth(168)
        left_w.setStyleSheet(f"background:{BG_HEADER}; border-right:1px solid {BORDER};")
        lh = QHBoxLayout(left_w); lh.setContentsMargins(6,4,6,4); lh.setSpacing(4)
        for icon, tip in [("💾","저장"),("📂","불러오기"),("◀","뒤로"),("▶","앞으로")]:
            lh.addWidget(self._icon_btn(icon, tip))
        h.addWidget(left_w)

        right_w = QWidget(); right_w.setStyleSheet(f"background:{BG_HEADER};")
        rh = QHBoxLayout(right_w); rh.setContentsMargins(8,4,8,0); rh.setSpacing(2)
        self.tab_btns = []
        for i, name in enumerate(TAB_NAMES):
            b = QPushButton(name); b.setFixedHeight(30); b.setMinimumWidth(80)
            b.setStyleSheet(_tab_btn_style(i == 0))
            b.clicked.connect(lambda _, idx=i: self._select(idx))
            self.tab_btns.append(b); rh.addWidget(b)
        rh.addStretch(); h.addWidget(right_w, stretch=1)

    @staticmethod
    def _icon_btn(icon, tip):
        b = QPushButton(icon); b.setFixedSize(28,28); b.setToolTip(tip)
        b.setStyleSheet(f"""
            QPushButton {{ background:{BG_BTN}; color:{TEXT_PRI};
                border:1px solid {BORDER_LT}; border-radius:4px; font-size:12px; }}
            QPushButton:hover {{ background:{BG_HOVER}; border-color:{BRAND}; }}
            QPushButton:pressed {{ background:{BRAND}20; }}
        """); return b

    def _select(self, idx: int):
        for i, b in enumerate(self.tab_btns): b.setStyleSheet(_tab_btn_style(i == idx))
        self.tab_changed.emit(idx)


# ══════════════════════════════════════════════════════════════════
# 사이드 패널
# ══════════════════════════════════════════════════════════════════
class SidePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(168); self.setStyleSheet(f"background:{BG_SIDE};")
        v = QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(0)

        self.title = QLabel("Run"); self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setFixedHeight(36)
        self.title.setStyleSheet(f"color:{BRAND}; font-size:12px; font-weight:700;"
                                 f" background:{BG_HEADER}; border-bottom:1px solid {BORDER};")
        v.addWidget(self.title)

        self.stack = QStackedWidget(); self.stack.setStyleSheet("background:transparent;")
        for items in SIDE_MENUS:
            page = QWidget(); page.setStyleSheet("background:transparent;")
            pv = QVBoxLayout(page); pv.setContentsMargins(0,4,0,4); pv.setSpacing(0)
            for (item,) in items:
                btn = QPushButton(item); btn.setFixedHeight(30)
                btn.setStyleSheet(f"""
                    QPushButton {{ text-align:left; padding-left:14px;
                        background:transparent; color:{TEXT_SEC};
                        border:none; border-bottom:1px solid {BORDER_LT}; font-size:10px; }}
                    QPushButton:hover {{ background:{BG_HOVER}; color:{BRAND}; }}
                """); pv.addWidget(btn)
            pv.addStretch(); self.stack.addWidget(page)
        v.addWidget(self.stack)

    def switch_tab(self, idx: int):
        self.title.setText(TAB_NAMES[idx]); self.stack.setCurrentIndex(idx)


# ══════════════════════════════════════════════════════════════════
# Run 탭
# ══════════════════════════════════════════════════════════════════
class RunPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG_CANVAS};")
        v = QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(0)

        bar = QWidget(); bar.setFixedHeight(32)
        bar.setStyleSheet(f"background:{BG_HEADER}; border-bottom:1px solid {BORDER_LT};")
        bh = QHBoxLayout(bar); bh.setContentsMargins(8,0,8,0); bh.setSpacing(6)
        for label in ["원본","검출 결과","깊이맵","포인트 클라우드"]:
            b = QPushButton(label); b.setFixedHeight(24); b.setMinimumWidth(72)
            b.setCheckable(True); b.setChecked(label == "검출 결과")
            b.setStyleSheet(f"""
                QPushButton {{ background:{BG_BTN}; color:{TEXT_SEC};
                    border:1px solid {BORDER_LT}; border-radius:3px;
                    font-size:10px; padding:0 8px; }}
                QPushButton:checked {{ background:{BRAND}22; color:{BRAND}; border-color:{BRAND}; }}
                QPushButton:hover {{ background:{BG_HOVER}; color:{BRAND}; }}
            """); bh.addWidget(b)
        bh.addStretch()
        self.info_lbl = QLabel("객체: —   신뢰도: —   포즈: —   fit: —")
        self.info_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        bh.addWidget(self.info_lbl); v.addWidget(bar)

        self.img_lbl = QLabel("비전 탐지 결과 이미지 출력창")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet(f"background:{BG_CANVAS}; color:{TEXT_DIM}; font-size:16px;")
        self.img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.img_lbl, stretch=1)

    def show_image(self, path: str):
        if not path or not os.path.exists(path): return
        pix = QPixmap(path)
        if pix.isNull(): return
        self.img_lbl.setPixmap(pix.scaled(self.img_lbl.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def show_result(self, result: dict):
        cls = result.get("class","—"); score = result.get("score","—")
        fitness = result.get("fitness","—"); tcp = result.get("tcp",[])
        tcp_str = f"[{', '.join(f'{v:.1f}' for v in tcp)}]" if tcp else "—"
        score_s = f"{score:.2f}" if isinstance(score, float) else str(score)
        fit_s   = f"{fitness:.4f}" if isinstance(fitness, float) else str(fitness)
        self.info_lbl.setText(f"객체: {cls}   신뢰도: {score_s}   TCP: {tcp_str}   fit: {fit_s}")
        self.info_lbl.setStyleSheet(f"color:{TEXT_PRI}; font-size:10px;")


# ══════════════════════════════════════════════════════════════════
# 플레이스홀더
# ══════════════════════════════════════════════════════════════════
def _make_placeholder(label: str) -> QWidget:
    page = QWidget(); page.setStyleSheet(f"background:{BG_WHITE};")
    v = QVBoxLayout(page); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
    bar = QWidget(); bar.setFixedHeight(32)
    bar.setStyleSheet(f"background:{BG_HEADER}; border-bottom:1px solid {BORDER_LT};")
    bh = QHBoxLayout(bar); bh.setContentsMargins(10,0,10,0)
    lbl = QLabel(label); lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; font-weight:600;")
    bh.addWidget(lbl); bh.addStretch(); v.addWidget(bar)
    body = QLabel(label); body.setAlignment(Qt.AlignmentFlag.AlignCenter)
    body.setStyleSheet(f"color:{TEXT_DIM}; font-size:16px; background:{BG_WHITE};")
    v.addWidget(body, stretch=1); return page


# ══════════════════════════════════════════════════════════════════
# Camera 탭
# ══════════════════════════════════════════════════════════════════
CAMERA_PRESETS = {
    "Orbbec Femto Bolt":      {"resolutions":["1280×720","640×576","320×288"],
                               "depth_min_mm":500,"depth_max_mm":5000,"exposure_us":8000,
                               "exposure_min":100,"exposure_max":100000},
    "LUCID Helios2 (ToF)":    {"resolutions":["640×480","320×240"],
                               "depth_min_mm":200,"depth_max_mm":6000,"exposure_us":1000,
                               "exposure_min":100,"exposure_max":10000},
    "Intel RealSense D435i":  {"resolutions":["1280×720","848×480","640×480","424×240"],
                               "depth_min_mm":105,"depth_max_mm":10000,"exposure_us":8500,
                               "exposure_min":1,"exposure_max":200000},
    "Microsoft Azure Kinect": {"resolutions":["1280×720 (NFOV)","320×288 (NFOV Binned)",
                                              "1024×1024 (WFOV)","512×512 (WFOV Binned)"],
                               "depth_min_mm":250,"depth_max_mm":5460,"exposure_us":16000,
                               "exposure_min":500,"exposure_max":1000000},
    "Custom / Other":         {"resolutions":["1280×720","640×480"],
                               "depth_min_mm":100,"depth_max_mm":10000,"exposure_us":10000,
                               "exposure_min":1,"exposure_max":1000000},
}

class CameraPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG_WHITE};")
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        bar = QWidget(); bar.setFixedHeight(32)
        bar.setStyleSheet(f"background:{BG_HEADER}; border-bottom:1px solid {BORDER_LT};")
        bh = QHBoxLayout(bar); bh.setContentsMargins(10,0,10,0)
        lbl = QLabel("Camera  설정 화면")
        lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; font-weight:600;")
        bh.addWidget(lbl); bh.addStretch(); root.addWidget(bar)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{BG_WHITE}; }}"
                             f"QScrollBar:vertical {{ background:{BG_APP}; width:6px; border:none; }}"
                             f"QScrollBar::handle:vertical {{ background:{BORDER_LT}; border-radius:3px; }}")

        body = QWidget()
        body.setStyleSheet(_field_style() + _group_style() + f"background:{BG_WHITE};")
        bv = QVBoxLayout(body); bv.setContentsMargins(32,24,32,32); bv.setSpacing(20)

        fs = _field_style(); gs = _group_style()

        grp_cam = QGroupBox("카메라 선택"); grp_cam.setStyleSheet(gs)
        gcl = QFormLayout(grp_cam); gcl.setContentsMargins(16,16,16,12); gcl.setSpacing(10)
        gcl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.cam_combo = QComboBox(); self.cam_combo.addItems(list(CAMERA_PRESETS.keys()))
        self.cam_combo.setFixedWidth(260); self.cam_combo.setStyleSheet(fs)
        self.cam_combo.currentTextChanged.connect(self._on_cam_changed)
        self.driver_lbl = QLabel(); self.driver_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        gcl.addRow(self._row_lbl("카메라 모델"), self.cam_combo)
        gcl.addRow("", self.driver_lbl); bv.addWidget(grp_cam)

        grp_res = QGroupBox("해상도"); grp_res.setStyleSheet(gs)
        grl = QFormLayout(grp_res); grl.setContentsMargins(16,16,16,12); grl.setSpacing(10)
        grl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.res_combo = QComboBox(); self.res_combo.setFixedWidth(220)
        self.res_combo.setStyleSheet(fs)
        grl.addRow(self._row_lbl("해상도"), self.res_combo); bv.addWidget(grp_res)

        grp_depth = QGroupBox("깊이 탐지 범위"); grp_depth.setStyleSheet(gs)
        gdl = QFormLayout(grp_depth); gdl.setContentsMargins(16,16,16,12); gdl.setSpacing(10)
        gdl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.depth_min = QSpinBox(); self.depth_min.setRange(1,50000)
        self.depth_min.setSuffix("  mm"); self.depth_min.setFixedWidth(140)
        self.depth_min.setStyleSheet(fs)
        self.depth_max = QSpinBox(); self.depth_max.setRange(1,50000)
        self.depth_max.setSuffix("  mm"); self.depth_max.setFixedWidth(140)
        self.depth_max.setStyleSheet(fs)
        depth_row = QWidget(); depth_row.setStyleSheet("background:transparent;")
        drh = QHBoxLayout(depth_row); drh.setContentsMargins(0,0,0,0); drh.setSpacing(10)
        for w in [QLabel("최솟값"), self.depth_min, QLabel("—"), QLabel("최댓값"), self.depth_max]:
            drh.addWidget(w)
        drh.addStretch()
        for l in depth_row.findChildren(QLabel):
            l.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px; background:transparent;")
        gdl.addRow(self._row_lbl("탐지 범위"), depth_row)
        gdl.addRow("", self._hint("카메라 사양 내에서 설정하세요. 범위를 좁힐수록 노이즈가 감소합니다."))
        bv.addWidget(grp_depth)

        grp_exp = QGroupBox("노출 시간"); grp_exp.setStyleSheet(gs)
        gel = QFormLayout(grp_exp); gel.setContentsMargins(16,16,16,12); gel.setSpacing(10)
        gel.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.exposure = QSpinBox(); self.exposure.setRange(1,1000000)
        self.exposure.setSuffix("  μs"); self.exposure.setFixedWidth(160)
        self.exposure.setStyleSheet(fs)
        exp_row = QWidget(); exp_row.setStyleSheet("background:transparent;")
        erh = QHBoxLayout(exp_row); erh.setContentsMargins(0,0,0,0); erh.setSpacing(10)
        erh.addWidget(self.exposure); erh.addStretch()
        gel.addRow(self._row_lbl("노출 시간"), exp_row)
        gel.addRow("", self._hint("값이 클수록 밝아지나 모션 블러가 증가합니다."))
        bv.addWidget(grp_exp)

        btn_row = QWidget(); btn_row.setStyleSheet("background:transparent;")
        brh = QHBoxLayout(btn_row); brh.setContentsMargins(0,0,0,0); brh.setSpacing(10)
        brh.addStretch()
        self.btn_reset = _apply_btn("초기값 복원", TEXT_DIM)
        self.btn_apply = _apply_btn("적용", BRAND)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_apply.clicked.connect(self._on_apply)
        brh.addWidget(self.btn_reset); brh.addWidget(self.btn_apply)
        bv.addWidget(btn_row); bv.addStretch()

        scroll.setWidget(body); root.addWidget(scroll, stretch=1)
        self._on_cam_changed(self.cam_combo.currentText())

    @staticmethod
    def _row_lbl(text):
        l = QLabel(text); l.setStyleSheet(f"color:{TEXT_PRI}; font-size:11px; font-weight:600;")
        l.setFixedWidth(90); return l

    @staticmethod
    def _hint(text):
        l = QLabel(text); l.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        l.setWordWrap(True); return l

    def _on_cam_changed(self, name: str):
        p = CAMERA_PRESETS.get(name, {})
        self.res_combo.clear(); self.res_combo.addItems(p.get("resolutions",[]))
        self.depth_min.setValue(p.get("depth_min_mm",100))
        self.depth_max.setValue(p.get("depth_max_mm",5000))
        self.exposure.setRange(p.get("exposure_min",1), p.get("exposure_max",1000000))
        self.exposure.setValue(p.get("exposure_us",10000))
        driver_map = {
            "Orbbec Femto Bolt":"드라이버: pyorbbecsdk2",
            "LUCID Helios2 (ToF)":"드라이버: lucid_sdk / arena_api",
            "Intel RealSense D435i":"드라이버: pyrealsense2",
            "Microsoft Azure Kinect":"드라이버: pyk4a",
            "Custom / Other":"드라이버: camera_driver.py 직접 구현",
        }
        self.driver_lbl.setText(driver_map.get(name,""))

    def _on_reset(self): self._on_cam_changed(self.cam_combo.currentText())
    def _on_apply(self): print("[CameraPage] 적용:", self.get_config())

    def get_config(self) -> dict:
        return {"camera": {
            "model": self.cam_combo.currentText(),
            "resolution": self.res_combo.currentText(),
            "depth_min_mm": self.depth_min.value(),
            "depth_max_mm": self.depth_max.value(),
            "exposure_us": self.exposure.value(),
        }}

    def set_config(self, cfg: dict):
        cam_cfg = cfg.get("camera",{})
        if cam_cfg.get("model","") in CAMERA_PRESETS:
            self.cam_combo.setCurrentText(cam_cfg["model"])
        if "resolution" in cam_cfg:
            idx = self.res_combo.findText(cam_cfg["resolution"])
            if idx >= 0: self.res_combo.setCurrentIndex(idx)
        if "depth_min_mm" in cam_cfg: self.depth_min.setValue(cam_cfg["depth_min_mm"])
        if "depth_max_mm" in cam_cfg: self.depth_max.setValue(cam_cfg["depth_max_mm"])
        if "exposure_us"  in cam_cfg: self.exposure.setValue(cam_cfg["exposure_us"])


# ══════════════════════════════════════════════════════════════════
# Pick Point 탭
# ══════════════════════════════════════════════════════════════════
_BOX_VERTS = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
_BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
_BOX_TRIS  = [
    (0,2,1),(0,3,2),   # -Z 면
    (4,5,6),(4,6,7),   # +Z 면
    (0,1,5),(0,5,4),   # -Y 면
    (2,3,7),(2,7,6),   # +Y 면
    (1,2,6),(1,6,5),   # +X 면
    (0,4,7),(0,7,3),   # -X 면
]

def _cyl_tris():
    tris = []
    N = 12
    # 옆면
    for i in range(N):
        a,b = i, (i+1)%N
        tris += [(a, b+N, b), (a, a+N, b+N)]
    # 위아래 캡
    for i in range(1, N-1):
        tris.append((0, i, i+1))
        tris.append((N, N+i+1, N+i))
    return tris

_CYL_TRIS = _cyl_tris()

_BRACKET_VERTS = [
    (-1.5,-0.2,-0.5),(1.5,-0.2,-0.5),(1.5,0.2,-0.5),(-1.5,0.2,-0.5),
    (-1.5,-0.2,0.5),(1.5,-0.2,0.5),(1.5,0.2,0.5),(-1.5,0.2,0.5),
    (-1.5,-0.2,-0.5),(-1.5,1.0,-0.5),(-1.5,1.0,0.5),(-1.5,-0.2,0.5),
    (1.5,-0.2,-0.5),(1.5,1.0,-0.5),(1.5,1.0,0.5),(1.5,-0.2,0.5),
]
_BRACKET_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7),
                  (8,9),(9,10),(10,11),(12,13),(13,14),(14,15)]
_BRACKET_TRIS  = [
    # 가로 바 6면
    (0,1,5),(0,5,4), (2,3,7),(2,7,6), (1,2,6),(1,6,5),
    (0,4,7),(0,7,3), (0,2,1),(0,3,2), (4,5,6),(4,6,7),
    # 좌측 세로 바
    (8,9,10),(8,10,11),
    # 우측 세로 바
    (12,14,13),(12,15,14),
]

_CYL_VERTS = (
    [(math.cos(a*math.pi/6),math.sin(a*math.pi/6),-1) for a in range(12)] +
    [(math.cos(a*math.pi/6),math.sin(a*math.pi/6), 1) for a in range(12)]
)
_CYL_EDGES = (
    [(i,(i+1)%12) for i in range(12)] +
    [(i+12,(i+1)%12+12) for i in range(12)] + [(i,i+12) for i in range(12)]
)

SHAPE_PRESETS = {
    "박스 (Box)":       (_BOX_VERTS,     _BOX_EDGES,     _BOX_TRIS),
    "원통 (Cylinder)":  (_CYL_VERTS,     _CYL_EDGES,     _CYL_TRIS),
    "브라켓 (Bracket)": (_BRACKET_VERTS, _BRACKET_EDGES, _BRACKET_TRIS),
}


def _project(verts, rx, ry, rz, scale, cx, cy):
    """Roll(Z) → Yaw(Y) → Pitch(X) 순 회전 후 등각투영."""
    cx_r,sx_r = math.cos(rx),math.sin(rx)
    cy_r,sy_r = math.cos(ry),math.sin(ry)
    cz_r,sz_r = math.cos(rz),math.sin(rz)
    pts = []
    for x,y,z in verts:
        x1 = x*cz_r - y*sz_r;  y1 = x*sz_r + y*cz_r
        x2 = x1*cy_r + z*sy_r; z2 = -x1*sy_r + z*cy_r
        y3 = y1*cx_r - z2*sx_r; z3 = y1*sx_r + z2*cx_r
        pts.append((cx + x2*scale, cy - y3*scale, z3))
    return pts


class ModelViewer(QWidget):
    vertex_picked = pyqtSignal(float, float, float)
    PICK_RADIUS = 18

    def __init__(self):
        super().__init__()
        self.setMinimumSize(380,320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background:{BG_CANVAS}; border:1px solid {BORDER_LT};")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._verts_norm = list(_BOX_VERTS); self._edges = list(_BOX_EDGES)
        self._tris: list = []        # 삼각형 인덱스 [(i,j,k), ...] — 면 음영용
        self._scale_norm = 1.0
        self._unit_to_m  = 1.0
        self._rx = math.radians(-23); self._ry = math.radians(29); self._rz = 0.0
        self._picked_idx    = -1
        self._pick_pos_norm = None

    def set_shape_norm(self, verts_norm, edges):
        self._verts_norm    = list(verts_norm); self._edges = list(edges)
        self._tris          = []
        self._scale_norm    = 1.0; self._unit_to_m = 1.0
        self._picked_idx    = -1
        self._pick_pos_norm = None
        self.update()

    def set_shape_real(self, verts_real, edges, tris, scale_norm, unit_to_m=1.0):
        """
        verts_real : 중심 이동만 한 원본 좌표 (파일 단위)
        edges      : [(i,j), ...] 와이어프레임용
        tris       : [(i,j,k), ...] 면 음영용 (없으면 [])
        scale_norm : 뷰어 정규화 스케일
        unit_to_m  : 단위 → m 변환 계수
        """
        self._verts_norm = [(x/scale_norm, y/scale_norm, z/scale_norm)
                            for x, y, z in verts_real]
        self._edges      = list(edges)
        self._tris       = list(tris)
        self._scale_norm = scale_norm
        self._unit_to_m  = unit_to_m
        self._picked_idx    = -1
        self._pick_pos_norm = None
        self.update()

    def set_pick_pos_mm(self, x_mm: float, y_mm: float, z_mm: float):
        """스핀박스 수동 입력 시 호출 — 십자선을 해당 좌표로 이동."""
        denom = self._scale_norm * self._unit_to_m
        if denom == 0:
            return
        self._pick_pos_norm = (x_mm / 1000.0 / denom,
                               y_mm / 1000.0 / denom,
                               z_mm / 1000.0 / denom)
        self._picked_idx = -1   # 꼭짓점 강조 해제 (자유 위치 모드)
        self.update()

    def set_rotation(self, rx_deg, ry_deg, rz_deg=0.0):
        self._rx = math.radians(rx_deg); self._ry = math.radians(ry_deg)
        self._rz = math.radians(rz_deg); self.update()

    def get_rotation_deg(self):
        return math.degrees(self._rx), math.degrees(self._ry), math.degrees(self._rz)

    def get_picked_real_m(self):
        """선택된 꼭짓점의 실제 좌표(m 단위). 미선택 시 None."""
        if self._picked_idx < 0: return None
        nx, ny, nz = self._verts_norm[self._picked_idx]
        # norm → 파일 원본 단위 → m
        x_m = nx * self._scale_norm * self._unit_to_m
        y_m = ny * self._scale_norm * self._unit_to_m
        z_m = nz * self._scale_norm * self._unit_to_m
        return (x_m, y_m, z_m)

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton: return
        mx,my = e.position().x(), e.position().y()
        W,H = self.width(), self.height()
        scale = min(W,H)*0.28
        pts = _project(self._verts_norm, self._rx, self._ry, self._rz, scale, W/2, H/2)
        best_idx, best_d = -1, float("inf")
        for i,(sx,sy,_) in enumerate(pts):
            d = math.hypot(sx-mx, sy-my)
            if d < best_d: best_d, best_idx = d, i
        if best_d > self.PICK_RADIUS: return
        self._picked_idx = best_idx
        # 클릭한 꼭짓점 좌표를 _pick_pos_norm에도 동기화
        self._pick_pos_norm = self._verts_norm[best_idx]
        self.update()
        rx, ry, rz = self.get_picked_real_m()
        self.vertex_picked.emit(rx, ry, rz)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H  = self.width(), self.height()
        scale = min(W, H) * 0.28
        cx, cy = W / 2, H / 2

        p.fillRect(self.rect(), QColor(BG_CANVAS))
        pts = _project(self._verts_norm, self._rx, self._ry, self._rz, scale, cx, cy)

        # ── 광원 방향 (카메라 시점 기준 — 좌상단 약간 앞) ────────────
        Lx, Ly, Lz = -0.4, -0.7, 1.0
        Llen = math.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
        Lx, Ly, Lz = Lx/Llen, Ly/Llen, Lz/Llen

        # ── 삼각형 면 렌더링 (Flat Shading) ─────────────────────────
        if self._tris:
            # 깊이 정렬 (Painter's Algorithm)
            tri_depths = []
            for ti, (i, j, k) in enumerate(self._tris):
                avg_z = (pts[i][2] + pts[j][2] + pts[k][2]) / 3
                tri_depths.append((avg_z, ti))
            tri_depths.sort()   # 가장 멀리 있는 면부터

            for avg_z, ti in tri_depths:
                i, j, k = self._tris[ti]
                ax, ay = pts[i][0], pts[i][1]
                bx, by = pts[j][0], pts[j][1]
                cx_, cy_ = pts[k][0], pts[k][1]

                # 법선 벡터 계산 (3D 공간에서)
                vx0 = self._verts_norm[i]; vx1 = self._verts_norm[j]; vx2 = self._verts_norm[k]
                ux = vx1[0]-vx0[0]; uy = vx1[1]-vx0[1]; uz = vx1[2]-vx0[2]
                wx = vx2[0]-vx0[0]; wy = vx2[1]-vx0[1]; wz = vx2[2]-vx0[2]
                nx = uy*wz - uz*wy; ny = uz*wx - ux*wz; nz = ux*wy - uy*wx
                nlen = math.sqrt(nx*nx + ny*ny + nz*nz)
                if nlen < 1e-10: continue
                nx, ny, nz = nx/nlen, ny/nlen, nz/nlen

                # 람베르트 음영: dot(N, L)
                dot = nx*Lx + ny*Ly + nz*Lz
                # 뒷면 컬백(Back-face): dot < 0 이면 뒷면
                is_back = dot < 0
                dot_abs = abs(dot)
                brightness = 0.18 + 0.62 * dot_abs   # 0.18 ~ 0.80

                if is_back:
                    # 뒷면: 어두운 회청색
                    r = int(30  * brightness + 20)
                    g = int(50  * brightness + 20)
                    b = int(80  * brightness + 30)
                else:
                    # 앞면: 밝은 청회색
                    r = int(100 * brightness + 60)
                    g = int(140 * brightness + 70)
                    b = int(200 * brightness + 30)

                face_col = QColor(min(r,255), min(g,255), min(b,255), 210)
                path = QPainterPath()
                path.moveTo(ax, ay); path.lineTo(bx, by)
                path.lineTo(cx_, cy_); path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(face_col); p.drawPath(path)

            # 와이어프레임 (얇게 — 면 위에 덧그림)
            pen = QPen(QColor(40, 80, 160, 80), 0.6)
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            for i, j in self._edges:
                p.drawLine(int(pts[i][0]), int(pts[i][1]),
                           int(pts[j][0]), int(pts[j][1]))

        else:
            # 삼각형 데이터 없음 → 기존 와이어프레임 모드
            for z, i, j in sorted(
                [((pts[i][2]+pts[j][2])/2, i, j) for i, j in self._edges]
            ):
                alpha = int(max(60, min(220, 140 + z * 40)))
                pen = QPen(QColor(40, 120, 200, alpha), 1.5)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen)
                p.drawLine(int(pts[i][0]), int(pts[i][1]),
                           int(pts[j][0]), int(pts[j][1]))

        # ── 꼭짓점 점 (클릭 대상) ────────────────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        for i, (sx, sy, _) in enumerate(pts):
            if i == self._picked_idx: continue
            # 면이 있을 때는 작게, 없을 때는 크게
            r = 3 if self._tris else 4
            p.setBrush(QColor(255, 220, 50, 200))
            p.drawEllipse(int(sx)-r, int(sy)-r, r*2, r*2)

        # ── 십자선 ───────────────────────────────────────────────────
        if self._picked_idx >= 0:
            sx, sy = int(pts[self._picked_idx][0]), int(pts[self._picked_idx][1])
            self._draw_crosshair(p, sx, sy)
        elif self._pick_pos_norm is not None:
            pp = _project([self._pick_pos_norm], self._rx, self._ry, self._rz,
                          scale, cx, cy)
            self._draw_crosshair(p, int(pp[0][0]), int(pp[0][1]))

        self._draw_axes_fixed(p, W-62, H-62, 44)

        p.setPen(QColor(TEXT_DIM)); p.setFont(QFont("sans-serif",9))
        has_pick = self._picked_idx >= 0 or self._pick_pos_norm is not None
        if not has_pick:
            guide = "꼭짓점을 클릭하면 픽포인트로 설정됩니다"
        elif self._picked_idx >= 0:
            rc = self.get_picked_real_m()
            guide = f"선택(꼭짓점):  X={rc[0]*1000:.2f}  Y={rc[1]*1000:.2f}  Z={rc[2]*1000:.2f}  mm"
        else:
            # 수동 입력 위치 — 정규화 → mm 역산
            nx,ny,nz = self._pick_pos_norm
            denom = self._scale_norm * self._unit_to_m
            x_mm = nx * denom * 1000
            y_mm = ny * denom * 1000
            z_mm = nz * denom * 1000
            guide = f"수동 지정:  X={x_mm:.2f}  Y={y_mm:.2f}  Z={z_mm:.2f}  mm"
        p.drawText(8, 18, guide); p.end()

    def _draw_crosshair(self, p: QPainter, sx: int, sy: int):
        r = 10
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(DANGER), 2))
        p.drawEllipse(sx-r, sy-r, r*2, r*2)
        p.setPen(QPen(QColor(DANGER), 1.5))
        p.drawLine(sx-r-6, sy, sx+r+6, sy)
        p.drawLine(sx, sy-r-6, sx, sy+r+6)
        p.setPen(QColor(DANGER))
        p.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        p.drawText(sx+r+4, sy-4, "Pick")

    def _draw_axes_fixed(self, p, ox, oy, length):
        dirs   = {"X":(1.0,0.0),"Y":(0.0,-1.0),"Z":(-0.6,0.6)}
        colors = {"X":QColor(220,60,60),"Y":QColor(60,180,60),"Z":QColor(60,120,220)}
        for lbl,(dx,dy) in dirs.items():
            col = colors[lbl]; ex,ey = ox+dx*length, oy+dy*length
            pen = QPen(col,2); pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen)
            p.drawLine(int(ox),int(oy),int(ex),int(ey))
            p.setFont(QFont("sans-serif",8,QFont.Weight.Bold)); p.setPen(col)
            p.drawText(int(ex)+2,int(ey)+4,lbl)


class PickPointPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG_WHITE};")
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        bar = QWidget(); bar.setFixedHeight(32)
        bar.setStyleSheet(f"background:{BG_HEADER}; border-bottom:1px solid {BORDER_LT};")
        bh = QHBoxLayout(bar); bh.setContentsMargins(10,0,10,0); bh.setSpacing(8)
        title_lbl = QLabel("Pick Point  설정")
        title_lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; font-weight:600;")
        bh.addWidget(title_lbl); bh.addStretch()
        self.status_lbl = QLabel("CAD 파일을 불러오거나 더미 형상을 선택하세요.")
        self.status_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        bh.addWidget(self.status_lbl); root.addWidget(bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background:{BORDER_LT}; }}")
        self.viewer = ModelViewer()
        self.viewer.vertex_picked.connect(self._on_vertex_picked)
        splitter.addWidget(self.viewer)

        # 로드된 원본 CAD 데이터 캐시 (축 매핑 실시간 재적용용)
        self._raw_verts: list = []
        self._raw_edges: list = []
        self._raw_tris:  list = []
        self._raw_scale: float = 1.0
        self._raw_unit:  float = 1.0

        panel = QScrollArea(); panel.setWidgetResizable(True); panel.setFixedWidth(320)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setStyleSheet(f"QScrollArea {{ border:none; background:{BG_SIDE}; }}"
                            f"QScrollBar:vertical {{ background:{BG_APP}; width:5px; border:none; }}"
                            f"QScrollBar::handle:vertical {{ background:{BORDER_LT}; border-radius:2px; }}")
        inner = QWidget(); inner.setStyleSheet(f"background:{BG_SIDE};")
        pv = QVBoxLayout(inner); pv.setContentsMargins(14,14,14,14); pv.setSpacing(14)
        fs = _field_style(); gs = _group_style()

        # 섹션 1: CAD 파일
        grp_cad = QGroupBox("CAD 파일"); grp_cad.setStyleSheet(gs)
        cv = QVBoxLayout(grp_cad); cv.setContentsMargins(12,14,12,12); cv.setSpacing(8)
        cv.addWidget(self._slbl("더미 형상 (테스트용)"))
        self.shape_combo = QComboBox(); self.shape_combo.addItems(list(SHAPE_PRESETS.keys()))
        self.shape_combo.setStyleSheet(fs)
        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)
        cv.addWidget(self.shape_combo)

        # 파일 단위 선택 (PLY/STL은 단위 정보가 없어 사용자가 지정해야 함)
        unit_row = QWidget(); unit_row.setStyleSheet("background:transparent;")
        uh = QHBoxLayout(unit_row); uh.setContentsMargins(0,0,0,0); uh.setSpacing(6)
        unit_lbl = QLabel("파일 단위:")
        unit_lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["mm  (밀리미터)", "m  (미터)", "cm  (센티미터)"])
        self.unit_combo.setStyleSheet(fs)
        uh.addWidget(unit_lbl); uh.addWidget(self.unit_combo); uh.addStretch()
        cv.addWidget(unit_row)
        btn_cad = _apply_btn("CAD 파일 불러오기  (.ply / .stl / .obj)", BRAND)
        btn_cad.clicked.connect(self._on_load_cad); cv.addWidget(btn_cad)
        self.cad_lbl = QLabel("불러온 파일: 없음")
        self.cad_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px;")
        self.cad_lbl.setWordWrap(True); cv.addWidget(self.cad_lbl)
        pv.addWidget(grp_cad)

        # 섹션 1-b: 좌표축 매핑 ─────────────────────────────────────
        # CAD 로컬 좌표계와 카메라 좌표계의 축 방향이 다를 때 맞춰주는 옵션
        grp_axis = QGroupBox("좌표축 매핑  (CAD → 카메라)")
        grp_axis.setStyleSheet(gs)
        av = QVBoxLayout(grp_axis); av.setContentsMargins(12,14,12,12); av.setSpacing(8)

        axis_hint = QLabel(
            "CAD 파일의 X/Y/Z 축이 카메라 좌표계와 다를 때 설정하세요.\n"
            "각 축에 어떤 CAD 축을 대응할지 선택합니다."
        )
        axis_hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px;"); axis_hint.setWordWrap(True)
        av.addWidget(axis_hint)

        AXIS_OPTIONS = ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]
        af = QFormLayout(); af.setSpacing(8)
        af.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def _axis_combo(default):
            c = QComboBox(); c.addItems(AXIS_OPTIONS)
            c.setCurrentText(default); c.setStyleSheet(fs); c.setFixedWidth(70)
            return c

        self.ax_x = _axis_combo("+X")
        self.ax_y = _axis_combo("+Y")
        self.ax_z = _axis_combo("+Z")

        # 축 매핑 변경 시 뷰어 실시간 갱신
        for combo in (self.ax_x, self.ax_y, self.ax_z):
            combo.currentTextChanged.connect(self._on_axis_changed)

        def _combo_row(combo, tip):
            w = QWidget(); w.setStyleSheet("background:transparent;")
            h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0); h.setSpacing(6)
            h.addWidget(combo)
            t = QLabel(tip); t.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px;")
            h.addWidget(t); h.addStretch(); return w

        af.addRow(self._plbl("Cam X ←"), _combo_row(self.ax_x, "카메라 X 방향"))
        af.addRow(self._plbl("Cam Y ←"), _combo_row(self.ax_y, "카메라 Y 방향"))
        af.addRow(self._plbl("Cam Z ←"), _combo_row(self.ax_z, "카메라 Z (깊이)"))
        av.addLayout(af)

        # 빠른 프리셋 버튼들
        preset_row = QWidget(); preset_row.setStyleSheet("background:transparent;")
        prh = QHBoxLayout(preset_row); prh.setContentsMargins(0,4,0,0); prh.setSpacing(4)
        preset_lbl = QLabel("프리셋:")
        preset_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px;")
        prh.addWidget(preset_lbl)

        PRESETS = [
            ("기본",      "+X", "+Y", "+Z"),
            ("Y↑Z↓",     "+X", "-Z", "+Y"),
            ("X↔Y",      "+Y", "+X", "+Z"),
            ("Z↑",       "+X", "+Z", "-Y"),
        ]
        for name, px, py, pz in PRESETS:
            b = QPushButton(name); b.setFixedHeight(22)
            b.setStyleSheet(f"""
                QPushButton {{ background:{BG_BTN}; color:{TEXT_SEC};
                    border:1px solid {BORDER_LT}; border-radius:3px; font-size:9px; padding:0 6px; }}
                QPushButton:hover {{ background:{BG_HOVER}; color:{BRAND}; border-color:{BRAND}; }}
            """)
            b.clicked.connect(lambda _, x=px, y=py, z=pz: self._apply_axis_preset(x,y,z))
            prh.addWidget(b)
        prh.addStretch()
        av.addWidget(preset_row)
        pv.addWidget(grp_axis)
        grp_rot = QGroupBox("물체 방향 설정"); grp_rot.setStyleSheet(gs)
        rl = QFormLayout(grp_rot); rl.setContentsMargins(12,14,12,12); rl.setSpacing(10)
        rl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sl_rx, self.lbl_rx = self._make_slider(-180,180,-23)
        self.sl_ry, self.lbl_ry = self._make_slider(-180,180, 29)
        self.sl_rz, self.lbl_rz = self._make_slider(-180,180,  0)
        for sl in (self.sl_rx, self.sl_ry, self.sl_rz):
            sl.valueChanged.connect(self._on_rotation_changed)
        for sp in (self.lbl_rx, self.lbl_ry, self.lbl_rz):
            sp.valueChanged.connect(self._on_rotation_changed)
        rl.addRow(self._plbl("Pitch (X)"), self._slider_row(self.sl_rx, self.lbl_rx))
        rl.addRow(self._plbl("Yaw   (Y)"), self._slider_row(self.sl_ry, self.lbl_ry))
        rl.addRow(self._plbl("Roll  (Z)"), self._slider_row(self.sl_rz, self.lbl_rz))
        btn_reset_view = _apply_btn("뷰 초기화", TEXT_DIM)
        btn_reset_view.clicked.connect(self._on_reset_view)
        rl.addRow("", btn_reset_view); pv.addWidget(grp_rot)

        # 섹션 3: CAD_PICK_LOCAL
        grp_local = QGroupBox("CAD_PICK_LOCAL  (클릭 또는 직접 입력)")
        grp_local.setStyleSheet(gs)
        lv = QVBoxLayout(grp_local); lv.setContentsMargins(12,14,12,12); lv.setSpacing(6)
        hint = QLabel("뷰어에서 꼭짓점을 클릭하거나 아래에 직접 입력하세요.\n단위: mm  (저장 시 m로 자동 변환)")
        hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px;"); hint.setWordWrap(True)
        lv.addWidget(hint)
        def _mspin(val=0.0):
            w = QDoubleSpinBox(); w.setRange(-100000.0,100000.0); w.setValue(val)
            w.setSuffix("  mm"); w.setDecimals(2); w.setStyleSheet(fs)
            w.setMinimumWidth(130); return w
        fl = QFormLayout(); fl.setSpacing(10); fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_lx = _mspin(); self.sp_ly = _mspin(); self.sp_lz = _mspin()
        fl.addRow(self._plbl("X"), self.sp_lx)
        fl.addRow(self._plbl("Y"), self.sp_ly)
        fl.addRow(self._plbl("Z"), self.sp_lz); lv.addLayout(fl)
        self.local_disp = QLabel("CAD_PICK_LOCAL = [0.0000, 0.0000, 0.0000, 1.0]  m")
        self.local_disp.setStyleSheet(f"color:{BRAND}; font-size:9px; font-weight:600;"
                                      f" font-family:monospace; padding-top:4px;")
        self.local_disp.setWordWrap(True); lv.addWidget(self.local_disp)
        for sp in (self.sp_lx, self.sp_ly, self.sp_lz):
            sp.valueChanged.connect(self._on_local_changed)
        pv.addWidget(grp_local)

        # 섹션 4: PICK_OFFSET
        grp_off = QGroupBox("PICK_OFFSET  (추가 미세 조정)"); grp_off.setStyleSheet(gs)
        of = QFormLayout(grp_off); of.setContentsMargins(12,14,12,12); of.setSpacing(10)
        of.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        def _ospin(val=0.0):
            w = QDoubleSpinBox(); w.setRange(-500.0,500.0); w.setValue(val)
            w.setSuffix("  mm"); w.setDecimals(1); w.setStyleSheet(fs)
            w.setMinimumWidth(130); return w
        self.sp_ox = _ospin(0.0); self.sp_oy = _ospin(0.0); self.sp_oz = _ospin(30.0)
        of.addRow(self._plbl("X offset"), self.sp_ox)
        of.addRow(self._plbl("Y offset"), self.sp_oy)
        of.addRow(self._plbl("Z offset"), self.sp_oz)
        off_hint = QLabel("PICK_OFFSET_X/Y/Z_MM — 그리퍼 TCP 보정용\n기존 코드 상수와 동일한 역할")
        off_hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px;"); off_hint.setWordWrap(True)
        of.addRow("", off_hint); pv.addWidget(grp_off)

        btn_save = _apply_btn("픽포인트 저장  (JSON)", BRAND)
        btn_save.clicked.connect(self._on_save)
        pv.addWidget(btn_save); pv.addStretch()

        panel.setWidget(inner); splitter.addWidget(panel)
        splitter.setSizes([640,320]); root.addWidget(splitter, stretch=1)

        self._on_shape_changed(self.shape_combo.currentText())
        self._on_rotation_changed()

    # ── 헬퍼 ────────────────────────────────────
    @staticmethod
    def _slbl(text):
        l = QLabel(text); l.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;"); return l

    @staticmethod
    def _plbl(text):
        l = QLabel(text); l.setStyleSheet(f"color:{TEXT_PRI}; font-size:10px; font-weight:600;")
        l.setFixedWidth(74); return l

    @staticmethod
    def _make_slider(lo, hi, val):
        sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(lo,hi); sl.setValue(val)
        sl.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height:4px; background:{BORDER_LT}; border-radius:2px; }}
            QSlider::handle:horizontal {{ width:14px; height:14px; margin:-5px 0;
                background:{BRAND}; border-radius:7px; }}
            QSlider::sub-page:horizontal {{ background:{BRAND}; border-radius:2px; }}
        """)
        sp = QSpinBox(); sp.setRange(lo,hi); sp.setValue(val); sp.setSuffix("°")
        sp.setFixedWidth(62)
        sp.setStyleSheet(f"""
            QSpinBox {{ background:{BG_WHITE}; color:{TEXT_PRI};
                border:1px solid {BORDER_LT}; border-radius:4px;
                padding:2px 4px; font-size:10px; }}
            QSpinBox:focus {{ border-color:{BRAND}; }}
            QSpinBox::up-button, QSpinBox::down-button {{ width:14px; border:none; background:{BG_BTN}; }}
        """)
        sl.valueChanged.connect(lambda v: (sp.blockSignals(True), sp.setValue(v), sp.blockSignals(False)))
        sp.valueChanged.connect(lambda v: (sl.blockSignals(True), sl.setValue(v), sl.blockSignals(False)))
        return sl, sp

    @staticmethod
    def _slider_row(sl, sp):
        w = QWidget(); w.setStyleSheet("background:transparent;")
        h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0); h.setSpacing(6)
        h.addWidget(sl); h.addWidget(sp); return w

    def _apply_axis_preset(self, px, py, pz):
        # blockSignals로 중복 호출 방지 후 한 번만 갱신
        for combo, val in zip((self.ax_x, self.ax_y, self.ax_z), (px, py, pz)):
            combo.blockSignals(True)
            combo.setCurrentText(val)
            combo.blockSignals(False)
        self._on_axis_changed()

    def _on_axis_changed(self):
        """축 매핑 콤보 변경 시 — 캐시된 원본 데이터에 재매핑 적용."""
        if not self._raw_verts:
            return   # 아직 파일 로드 전 (더미 프리셋은 매핑 불필요)
        remapped = self._remap_verts(self._raw_verts)
        self.viewer.set_shape_real(
            remapped, self._raw_edges, self._raw_tris,
            self._raw_scale, self._raw_unit
        )
        self.status_lbl.setText(
            f"축 매핑 적용:  Cam X←{self.ax_x.currentText()}"
            f"  Y←{self.ax_y.currentText()}"
            f"  Z←{self.ax_z.currentText()}"
        )

    def _remap_verts(self, verts: list) -> list:
        """
        축 매핑 콤보 선택에 따라 꼭짓점 좌표를 재배열.
        ax_x="+Y" 이면 카메라 X = CAD Y 값을 사용.
        """
        axis_map = {"+X":(0, 1),"+Y":(1, 1),"+Z":(2, 1),
                    "-X":(0,-1),"-Y":(1,-1),"-Z":(2,-1)}
        xi,xs = axis_map.get(self.ax_x.currentText(),(0,1))
        yi,ys = axis_map.get(self.ax_y.currentText(),(1,1))
        zi,zs = axis_map.get(self.ax_z.currentText(),(2,1))
        return [(v[xi]*xs, v[yi]*ys, v[zi]*zs) for v in verts]

    # ── 슬롯 ────────────────────────────────────
    def _on_shape_changed(self, name: str):
        preset = SHAPE_PRESETS.get(name)
        if preset:
            verts, edges, tris = preset
        else:
            verts, edges, tris = _BOX_VERTS, _BOX_EDGES, _BOX_TRIS
        # 더미 프리셋은 정규화 좌표 그대로 → set_shape_real에 scale=1, unit=1 로 넘김
        self.viewer.set_shape_real(verts, edges, tris, 1.0, 1.0)
        self.status_lbl.setText(f"더미 형상: {name}  ·  꼭짓점을 클릭해 픽포인트를 지정하세요.")
        self.cad_lbl.setText("불러온 파일: 없음")

    def _on_load_cad(self):
        path, _ = QFileDialog.getOpenFileName(self, "CAD 파일 선택", "",
                                              "Mesh/PointCloud (*.ply *.stl *.obj);;모든 파일 (*)")
        if not path: return
        fname = os.path.basename(path)
        self.status_lbl.setText(f"'{fname}'  읽는 중..."); QApplication.processEvents()
        try:
            import open3d as o3d, numpy as np
            ext = os.path.splitext(path)[1].lower()

            # 단위 → m 변환 계수
            unit_text = self.unit_combo.currentText()
            unit_to_m = {"mm": 0.001, "m": 1.0, "cm": 0.01}.get(unit_text[:2].strip(), 0.001)

            if ext == ".ply":
                mesh = o3d.io.read_triangle_mesh(path)
                geo_type = "mesh" if len(mesh.triangles) > 0 else "pcd"
                if geo_type == "pcd": pcd = o3d.io.read_point_cloud(path)
            elif ext in (".stl",".obj"):
                mesh = o3d.io.read_triangle_mesh(path); geo_type = "mesh"
            else: raise ValueError(f"지원하지 않는 형식: {ext}")

            if geo_type == "mesh":
                verts_np = np.asarray(mesh.vertices); tris_np = np.asarray(mesh.triangles)
                if len(verts_np) == 0: raise ValueError("꼭짓점이 없습니다.")
                center = verts_np.mean(axis=0); verts_c = verts_np - center
                scale_norm = float(np.abs(verts_c).max()) or 1.0

                # 삼각형이 너무 많으면 Open3D simplify로 줄임 (뷰어 성능)
                MAX_TRIS = 3000
                if len(tris_np) > MAX_TRIS:
                    ratio = MAX_TRIS / len(tris_np)
                    mesh_s = mesh.simplify_quadric_decimation(
                        int(len(tris_np) * ratio))
                    verts_np = np.asarray(mesh_s.vertices)
                    tris_np  = np.asarray(mesh_s.triangles)
                    center   = verts_np.mean(axis=0)
                    verts_c  = verts_np - center
                    scale_norm = float(np.abs(verts_c).max()) or 1.0

                # 엣지 (와이어프레임용)
                edge_set = set()
                for tri in tris_np:
                    for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
                        edge_set.add((min(a,b),max(a,b)))
                edges = list(edge_set)

                verts_real = [tuple(v) for v in verts_c]
                tris_list  = [tuple(t) for t in tris_np]
                size_mm = scale_norm * unit_to_m * 1000
                info = (f"메시  꼭짓점 {len(verts_real):,}  삼각형 {len(tris_list):,}"
                        f"  |  최대 반경 {size_mm:.1f} mm")
            else:
                pts_np = np.asarray(pcd.points)
                if len(pts_np) == 0: raise ValueError("포인트가 없습니다.")
                center = pts_np.mean(axis=0); pts_c = pts_np - center
                scale_norm = float(np.abs(pts_c).max()) or 1.0
                pcd_n = o3d.geometry.PointCloud()
                pcd_n.points = o3d.utility.Vector3dVector(pts_c)
                hull,_ = pcd_n.compute_convex_hull()
                verts_np2 = np.asarray(hull.vertices); tris_np2 = np.asarray(hull.triangles)
                edge_set = set()
                for tri in tris_np2:
                    for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
                        edge_set.add((min(a,b),max(a,b)))
                edges = list(edge_set); verts_real = [tuple(v) for v in verts_np2]
                tris_list = [tuple(t) for t in tris_np2]
                size_mm = scale_norm * unit_to_m * 1000
                info = (f"포인트클라우드 → 볼록껍질  꼭짓점 {len(verts_real)}"
                        f"  삼각형 {len(tris_list)}  |  최대 반경 {size_mm:.1f} mm")

            # 원본 캐시 저장 (축 매핑 실시간 재적용용)
            self._raw_verts = verts_real
            self._raw_edges = edges
            self._raw_tris  = tris_list
            self._raw_scale = scale_norm
            self._raw_unit  = unit_to_m

            remapped = self._remap_verts(verts_real)
            self.viewer.set_shape_real(remapped, edges, tris_list, scale_norm, unit_to_m)
            self.cad_lbl.setText(f"파일: {fname}  [{unit_text[:2].strip()}]")
            self.status_lbl.setText(info + "  ·  꼭짓점 클릭으로 픽포인트 지정")
            self.shape_combo.blockSignals(True); self.shape_combo.setCurrentIndex(-1)
            self.shape_combo.blockSignals(False)
        except ImportError:
            self.status_lbl.setText("open3d 미설치 — pip install open3d")
        except Exception as e:
            self.status_lbl.setText(f"로드 실패: {e}"); self.cad_lbl.setText(f"오류: {fname}")

    def _on_rotation_changed(self):
        rx = self.sl_rx.value(); ry = self.sl_ry.value(); rz = self.sl_rz.value()
        self.lbl_rx.blockSignals(True); self.lbl_rx.setValue(rx); self.lbl_rx.blockSignals(False)
        self.lbl_ry.blockSignals(True); self.lbl_ry.setValue(ry); self.lbl_ry.blockSignals(False)
        self.lbl_rz.blockSignals(True); self.lbl_rz.setValue(rz); self.lbl_rz.blockSignals(False)
        self.viewer.set_rotation(rx, ry, rz)

    def _on_reset_view(self):
        self.sl_rx.setValue(-23); self.sl_ry.setValue(29); self.sl_rz.setValue(0)

    def _on_vertex_picked(self, x_m, y_m, z_m):
        x_mm,y_mm,z_mm = x_m*1000, y_m*1000, z_m*1000
        for sp,v in zip((self.sp_lx,self.sp_ly,self.sp_lz),(x_mm,y_mm,z_mm)):
            sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
        self._refresh_local_disp(x_mm, y_mm, z_mm)
        self.status_lbl.setText(f"픽포인트 지정:  [{x_mm:.2f},  {y_mm:.2f},  {z_mm:.2f}]  mm")

    def _on_local_changed(self):
        x_mm = self.sp_lx.value()
        y_mm = self.sp_ly.value()
        z_mm = self.sp_lz.value()
        self.viewer.set_pick_pos_mm(x_mm, y_mm, z_mm)
        self._refresh_local_disp(x_mm, y_mm, z_mm)

    def _refresh_local_disp(self, x_mm, y_mm, z_mm):
        x_m,y_m,z_m = x_mm/1000, y_mm/1000, z_mm/1000
        self.local_disp.setText(f"CAD_PICK_LOCAL = [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, 1.0]  m")

    def _on_save(self):
        path,_ = QFileDialog.getSaveFileName(self,"픽포인트 저장","pickpoint.json","JSON (*.json)")
        if not path: return
        import json
        rx_d,ry_d,rz_d = self.viewer.get_rotation_deg()
        x_m,y_m,z_m = self.sp_lx.value()/1000, self.sp_ly.value()/1000, self.sp_lz.value()/1000
        data = {"CAD_PICK_LOCAL":[round(x_m,6),round(y_m,6),round(z_m,6),1.0],
                "PICK_OFFSET_X_MM":round(self.sp_ox.value(),2),
                "PICK_OFFSET_Y_MM":round(self.sp_oy.value(),2),
                "PICK_OFFSET_Z_MM":round(self.sp_oz.value(),2),
                "view":{"pitch_deg":round(rx_d,1),"yaw_deg":round(ry_d,1),
                        "roll_deg":round(rz_d,1),"cad_file":self.cad_lbl.text()}}
        with open(path,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        self.status_lbl.setText(f"저장 완료:  {os.path.basename(path)}")

    def get_config(self) -> dict:
        rx_d,ry_d,rz_d = self.viewer.get_rotation_deg()
        x_m,y_m,z_m = self.sp_lx.value()/1000, self.sp_ly.value()/1000, self.sp_lz.value()/1000
        return {"CAD_PICK_LOCAL":[round(x_m,6),round(y_m,6),round(z_m,6),1.0],
                "PICK_OFFSET_X_MM":round(self.sp_ox.value(),2),
                "PICK_OFFSET_Y_MM":round(self.sp_oy.value(),2),
                "PICK_OFFSET_Z_MM":round(self.sp_oz.value(),2)}


# ══════════════════════════════════════════════════════════════════
# 중앙 스택
# ══════════════════════════════════════════════════════════════════
class CenterStack(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.run_page = RunPage();          self.addWidget(self.run_page)
        self.addWidget(_make_placeholder("Detection  설정 화면"))
        self.cam_page = CameraPage();      self.addWidget(self.cam_page)
        self.addWidget(_make_placeholder("Calibration  설정 화면"))
        self.pick_page = PickPointPage();  self.addWidget(self.pick_page)
        for name in TAB_NAMES[5:]:
            self.addWidget(_make_placeholder(f"{name}  설정 화면"))

    def switch_tab(self, idx: int): self.setCurrentIndex(idx)
    def show_image(self, path: str): self.run_page.show_image(path)
    def show_result(self, result: dict): self.run_page.show_result(result)


# ══════════════════════════════════════════════════════════════════
# 로그 패널
# ══════════════════════════════════════════════════════════════════
class LogPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(150); self.setStyleSheet(f"background:{BG_LOG};")
        v = QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(26)
        hdr.setStyleSheet(f"background:{BG_HEADER}; border-bottom:1px solid {BORDER_LT};")
        hh = QHBoxLayout(hdr); hh.setContentsMargins(10,0,10,0)
        lbl = QLabel("로그 출력화면")
        lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; font-weight:700;")
        hh.addWidget(lbl); hh.addStretch()
        clr = QPushButton("초기화"); clr.setFixedSize(48,18)
        clr.setStyleSheet(f"""
            QPushButton {{ background:{BG_BTN}; color:{TEXT_DIM};
                border:1px solid {BORDER_LT}; border-radius:3px; font-size:9px; }}
            QPushButton:hover {{ color:{DANGER}; border-color:{DANGER}; }}
        """)
        hh.addWidget(clr); v.addWidget(hdr)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(f"""
            QTextEdit {{ background:{BG_LOG}; color:{TEXT_LOG};
                font-family:'Monospace','Courier New','D2Coding';
                font-size:11px; border:none; padding:6px 10px; }}
            QScrollBar:vertical {{ background:{BG_APP}; width:6px; border:none; }}
            QScrollBar::handle:vertical {{ background:{BORDER_LT}; border-radius:3px; }}
        """)
        clr.clicked.connect(self.log_box.clear); v.addWidget(self.log_box)
        self.push("시스템 초기화 완료.", "INFO")
        self.push("UIBridge 대기 중 — PipelineRunner 미연결.", "WARN")

    def push(self, msg: str, level: str = "INFO"):
        col = {"INFO":TEXT_SEC,"OK":SUCCESS,"WARN":WARN,"ERR":DANGER}.get(level, TEXT_SEC)
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(
            f'<span style="color:{TEXT_DIM}">[{ts}]</span> '
            f'<span style="color:{col}">[{level}]</span> '
            f'<span style="color:{TEXT_LOG}">{msg}</span>'
        )
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())


# ══════════════════════════════════════════════════════════════════
# 메인 윈도우
# ══════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BENIROBO — 3D Vision Robot")
        self.setMinimumSize(960,640); self.resize(1200,740)
        self.running = False; self._build_ui()
        self.bridge = UIBridge(self); self.bridge.connect_signals()

    def _build_ui(self):
        self.setStyleSheet(f"""
            * {{ font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',sans-serif; }}
            QMainWindow, QWidget {{ background:{BG_APP}; color:{TEXT_PRI}; }}
            QSplitter::handle {{ background:{BORDER}; }}
        """)
        root_w = QWidget(); self.setCentralWidget(root_w)
        root_v = QVBoxLayout(root_w); root_v.setContentsMargins(0,0,0,0); root_v.setSpacing(0)

        self.title_bar = TitleBar(); root_v.addWidget(self.title_bar); root_v.addWidget(hline())
        self.sub_bar = SubToolBar(); root_v.addWidget(self.sub_bar); root_v.addWidget(hline())

        body_h = QHBoxLayout(); body_h.setContentsMargins(0,0,0,0); body_h.setSpacing(0)
        self.side = SidePanel(); body_h.addWidget(self.side); body_h.addWidget(vline())

        right_v = QVBoxLayout(); right_v.setContentsMargins(0,0,0,0); right_v.setSpacing(0)
        self.center = CenterStack(); right_v.addWidget(self.center, stretch=1)
        right_v.addWidget(hline())
        self.log = LogPanel(); right_v.addWidget(self.log)
        right_w = QWidget(); right_w.setLayout(right_v)
        body_h.addWidget(right_w, stretch=1)
        body_w = QWidget(); body_w.setLayout(body_h); root_v.addWidget(body_w, stretch=1)

        sb = QStatusBar(); sb.setFixedHeight(22)
        sb.setStyleSheet(f"background:{BG_HEADER}; color:{TEXT_DIM};"
                         f" font-size:10px; border-top:1px solid {BORDER};")
        self.setStatusBar(sb)
        self.st_lbl = QLabel("준비"); sb.addWidget(self.st_lbl)
        sb.addPermanentWidget(QLabel("BENIROBO Vision System  |  Ubuntu 22.04  |  Open3D 0.18  |  PyQt6"))

        self.sub_bar.tab_changed.connect(self.side.switch_tab)
        self.sub_bar.tab_changed.connect(self.center.switch_tab)
        self.sub_bar.tab_changed.connect(self._on_tab_changed)

        run_page = self.side.stack.widget(0)
        btns = [run_page.layout().itemAt(i).widget()
                for i in range(run_page.layout().count())
                if run_page.layout().itemAt(i).widget()]
        labels  = ["파이프라인 시작","Step 실행","일시정지","중지"]
        actions = [self._on_run, self._on_step, self._on_pause, self._on_stop]
        for btn,lbl,act in zip(btns,labels,actions):
            if hasattr(btn,'text') and btn.text() in labels: btn.clicked.connect(act)

    def _on_tab_changed(self, idx: int):
        self.st_lbl.setText(TAB_NAMES[idx]); self.st_lbl.setStyleSheet(f"color:{TEXT_PRI}; font-size:10px;")

    def _on_run(self):
        if self.running: return
        self.running = True
        self.st_lbl.setText("● 실행 중"); self.st_lbl.setStyleSheet(f"color:{DANGER}; font-size:10px;")
        self.title_bar.set_indicator("RUNNING", True); self.bridge.start_pipeline()

    def _on_step(self): self.log.push("Step 실행 요청", "INFO")
    def _on_pause(self): self.log.push("일시정지 요청", "INFO")

    def _on_stop(self):
        if not self.running: return
        self.bridge.stop_pipeline(); self._set_idle()

    def _set_idle(self):
        self.running = False
        self.st_lbl.setText("● 대기 중"); self.st_lbl.setStyleSheet(f"color:{SUCCESS}; font-size:10px;")
        self.title_bar.set_indicator("RUNNING", False)

    def on_image(self, path: str):   self.center.show_image(path)
    def on_result(self, result: dict):
        self.center.show_result(result)
        cls = result.get("class","—"); fit = result.get("fitness","—")
        fit_s = f"{fit:.4f}" if isinstance(fit, float) else str(fit)
        self.st_lbl.setText(f"클래스: {cls}  |  fitness: {fit_s}")
        self.st_lbl.setStyleSheet(f"color:{TEXT_PRI}; font-size:10px;")

    def on_pipeline_finished(self):
        self.log.push("파이프라인 완료.", "OK"); self._set_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = MainWindow(); win.show(); sys.exit(app.exec())