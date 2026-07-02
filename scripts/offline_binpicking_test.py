"""오프라인 빈피킹 파이프라인 (카메라 없음)
============================================================
5_Run_binpicking_TCP.py 에서 카메라·TCP 부분만 제거하고,
사전 저장된 파일로 동일한 Detection → ICP → 결과 포맷을 재현합니다.

입력 디렉토리 구조 (--input 경로):
    <input>/                                   ← 예) data/captures/live
        intensity/               *.png          grayscale intensity 이미지
        pointcloud_organized/    frame_XXXX.npy H×W×3 float32, mm 단위
        valid_mask/              frame_XXXX.npy H×W bool

  ※ 5_Run_binpicking_TCP.py 가 --out 으로 저장하는 폴더 구조와 완전히 동일합니다.
     TCP 스크립트의 --out 경로를 그대로 --input 에 넣으면 됩니다.

사용법:
    # 자동 모드 (입력 디렉토리에서 최신 파일 세트 자동 선택)
    python offline_binpicking_test.py

    # 입력 경로 직접 지정 (TCP 저장 폴더를 바로 사용)
    python offline_binpicking_test.py --input data/captures/live

    # 파일 직접 지정
    python offline_binpicking_test.py \\
        --intensity  data/captures/live/intensity/20250626_120000.png \\
        --pcd_npy    data/captures/live/pointcloud_organized/frame_0001.npy \\
        --mask_npy   data/captures/live/valid_mask/frame_0001.npy

    # 대화형 모드 (목록에서 골라가며 반복)
    python offline_binpicking_test.py --interactive

출력 (--out/results/):
    frame_XXXX_overlay.png
    frame_XXXX_colored.ply
    frame_XXXX_result.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d 필요. pip install open3d", flush=True)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import RTMDetInferencer  # noqa: E402


# =============================================================================
# 설정 — 5_Run_binpicking_TCP.py 와 완전 동일하게 유지
# =============================================================================

WORK_DIR    = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1"
CONFIG_PATH = WORK_DIR / "rtmdet-ins_bracket.py"

_candidates = sorted(WORK_DIR.glob("best_*.pth"))
if not _candidates:
    print(f"ERROR: best 모델이 없습니다: {WORK_DIR}", flush=True)
    sys.exit(1)
CHECKPOINT_PATH = _candidates[-1]

SCORE_THRESHOLD         = 0.3
MIN_POINTS_PER_INSTANCE = 100
MASK_IOU_THRESHOLD      = 0.6

CAD_PATH = ROOT / "data" / "cad" / "bracket_v2.stl"

CAD_SAMPLE_POINTS = 20000
VOXEL_SIZE_CAD    = 0.002
VOXEL_SIZE_SCENE  = 0.003

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

ICP_STAGES = [
    {"max_dist": 0.020, "max_iter": 100},
    {"max_dist": 0.010, "max_iter": 100},
    {"max_dist": 0.005, "max_iter": 100},
]
ICP_FITNESS_THRESHOLD = 0.5
XYZ_MAX_M             = 2.0

CAD_AXIS_CORRECTION_DEG = (-90, 90, 90)

CAD_PICK_LOCAL = np.array([0.000, -0.100, 0.031, 1.0])

PICK_OFFSET_X_MM = -5.0
PICK_OFFSET_Y_MM =  0.0
PICK_OFFSET_Z_MM =  0.0

_PALETTE_BGR = np.array([
    [ 50,  50, 255], [ 50, 200,  50], [255, 100,  50],
    [ 30, 180, 255], [230,  50, 180], [200, 200,  30],
], dtype=np.uint8)
_PALETTE_RGB_FLOAT = _PALETTE_BGR[:, ::-1].astype(np.float64) / 255.0
_BG_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float64)



# =============================================================================
# ★ 카메라 대체: 저장 파일 로드 함수들
# =============================================================================

def load_frame_from_files(
    intensity_path: Path,
    pcd_npy_path: Path,
    mask_npy_path: Path,
):
    """
    cam.capture() 를 대체하는 함수.

    반환:
        gray          (H, W)     uint8   intensity 이미지
        pcd_organized (H, W, 3)  float32 XYZ 포인트, mm 단위
        valid_mask    (H, W)     bool    유효 픽셀 마스크
    """
    # intensity PNG → grayscale
    img = cv2.imread(str(intensity_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"intensity 이미지 로드 실패: {intensity_path}")
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # pointcloud_organized .npy → (H, W, 3) float32
    pcd_organized = np.load(str(pcd_npy_path)).astype(np.float32)

    # valid_mask .npy → (H, W) bool
    valid_mask = np.load(str(mask_npy_path)).astype(bool)

    print(f"  [로드] intensity : {intensity_path.name}  shape={gray.shape}", flush=True)
    print(f"  [로드] pcd_org   : {pcd_npy_path.name}  shape={pcd_organized.shape}", flush=True)
    print(f"  [로드] valid     : {mask_npy_path.name}"
          f"  valid={valid_mask.sum():,} / {valid_mask.size:,}", flush=True)

    return gray, pcd_organized, valid_mask


def find_latest_set(input_dir: Path):
    """
    input_dir 안의 서브폴더 구조에서 가장 최신 파일 세트를 자동 탐색.

    5_Run_binpicking_TCP.py 의 save_capture() 가 저장하는 구조:
        <input_dir>/
            intensity/               날짜시간.png       예) 20260530_161707.png
            pointcloud_organized/    frame_XXXX.npy     예) frame_0010.npy
            valid_mask/              frame_XXXX.npy     예) frame_0010.npy
    """
    pcd_dir       = input_dir / "pointcloud_organized"
    mask_dir      = input_dir / "valid_mask"
    intensity_dir = input_dir / "intensity"

    for d, label in [(pcd_dir, "pointcloud_organized"),
                     (mask_dir, "valid_mask"),
                     (intensity_dir, "intensity")]:
        if not d.exists():
            raise FileNotFoundError(
                f"서브폴더 없음: {d}\n"
                f"  → TCP 스크립트가 저장한 폴더를 --input 에 지정하세요.")

    # pcd 기준으로 최신 frame 번호 결정
    pcds = sorted(
        pcd_dir.glob("frame_*.npy"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not pcds:
        raise FileNotFoundError(f"frame_XXXX.npy 없음: {pcd_dir}")

    pcd_npy = pcds[0]
    num     = pcd_npy.stem.split("_")[-1]   # "0010"

    # valid_mask 는 같은 이름 (frame_XXXX.npy)
    mask_npy = mask_dir / f"frame_{num}.npy"
    if not mask_npy.exists():
        raise FileNotFoundError(f"valid_mask/frame_{num}.npy 없음: {mask_dir}")

    # intensity: 가장 최신 PNG
    pngs = sorted(
        intensity_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not pngs:
        raise FileNotFoundError(f"intensity PNG 없음: {intensity_dir}")

    return pngs[0], pcd_npy, mask_npy


def pick_set_interactive(input_dir: Path):
    """
    input_dir 안의 서브폴더 파일 목록을 보여주고 세트를 선택하게 함.
    5_Run_binpicking_TCP.py 가 저장하는 서브폴더 구조를 사용.
    """
    pcd_dir       = input_dir / "pointcloud_organized"
    mask_dir      = input_dir / "valid_mask"
    intensity_dir = input_dir / "intensity"

    for d, label in [(pcd_dir, "pointcloud_organized"),
                     (mask_dir, "valid_mask"),
                     (intensity_dir, "intensity")]:
        if not d.exists():
            raise FileNotFoundError(
                f"서브폴더 없음: {d}\n"
                f"  → TCP 스크립트가 저장한 폴더를 --input 에 지정하세요.")

    pcds = sorted(pcd_dir.glob("frame_*.npy"), key=lambda p: p.name)
    pngs = sorted(intensity_dir.glob("*.png"),  key=lambda p: p.name)

    if not pcds:
        raise FileNotFoundError(f"frame_XXXX.npy 없음: {pcd_dir}")

    print("\n" + "─" * 50)
    print(f"  입력 디렉토리: {input_dir}")
    print("─" * 50)

    # intensity PNG 선택
    if len(pngs) == 1:
        png = pngs[0]
        print(f"  intensity : {png.name} (자동 선택)")
    else:
        print("\n[intensity PNG 선택]")
        for i, p in enumerate(pngs):
            print(f"  {i+1:3d}. {p.name}")
        while True:
            try:
                raw = input(f"번호 입력 (1~{len(pngs)}, 기본=최신): ").strip()
                idx = int(raw) if raw else len(pngs)
                if 1 <= idx <= len(pngs):
                    png = pngs[idx - 1]
                    break
            except (ValueError, EOFError):
                pass

    # pcd / mask 세트 선택
    if len(pcds) == 1:
        pcd_npy = pcds[0]
        print(f"  pcd+mask  : {pcd_npy.name} (자동 선택)")
    else:
        print("\n[포인트클라우드 선택]")
        for i, p in enumerate(pcds):
            print(f"  {i+1:3d}. {p.name}")
        while True:
            try:
                raw = input(f"번호 입력 (1~{len(pcds)}, 기본=최신): ").strip()
                idx = int(raw) if raw else len(pcds)
                if 1 <= idx <= len(pcds):
                    pcd_npy = pcds[idx - 1]
                    break
            except (ValueError, EOFError):
                pass

    num      = pcd_npy.stem.split("_")[-1]
    mask_npy = mask_dir / f"frame_{num}.npy"
    if not mask_npy.exists():
        raise FileNotFoundError(f"valid_mask/frame_{num}.npy 없음: {mask_dir}")

    print(f"\n  선택됨:")
    print(f"    intensity : {png.name}")
    print(f"    pcd       : {pcd_npy.name}")
    print(f"    mask      : {mask_npy.name}")
    return png, pcd_npy, mask_npy


# =============================================================================
# Detection / ICP / 시각화 — 5_Run_binpicking_TCP.py 와 완전 동일
# =============================================================================

def overlay_results(image_bgr, results, valid_mask=None):
    overlay = image_bgr.copy()
    if valid_mask is not None:
        overlay[~valid_mask] = (overlay[~valid_mask] * 0.4).astype(np.uint8)
    for i, r in enumerate(results):
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        layer = np.zeros_like(overlay)
        layer[r.mask] = color
        overlay[r.mask] = (0.5 * overlay[r.mask] + 0.5 * layer[r.mask]).astype(np.uint8)
    for i, r in enumerate(results):
        color = tuple(int(c) for c in _PALETTE_BGR[i % len(_PALETTE_BGR)])
        x1, y1, x2, y2 = r.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"#{i} {r.class_name} {r.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def draw_picks_on_overlay(image_bgr: np.ndarray, picks_2d: list) -> np.ndarray:
    """overlay 이미지에 bbox, 픽포인트 좌표, ICP 매칭 점수를 추가.

    picks_2d 원소: (px, py, pick, icp_fitness, bbox)
        px, py      : 픽포인트 2D 위치 (bbox 중심)
        pick        : compute_pick_point() 반환 dict
        icp_fitness : ICP 정합 점수 0~1
        bbox        : [x1, y1, x2, y2]
    """
    out = image_bgr.copy()
    H, W = out.shape[:2]

    for i, (px, py, pick, icp_fitness, bbox) in enumerate(picks_2d):
        color  = tuple(int(c) for c in _PALETTE_BGR[i % len(_PALETTE_BGR)])
        pp     = pick["position_mm"]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # ── 마스킹 영역 사각형 ──────────────────────────────────────────────
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # ── 픽포인트 십자선 마커 ────────────────────────────────────────────
        cv2.drawMarker(out, (int(px), int(py)), color,
                       cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

        # ── 텍스트 2줄: 좌표 + ICP 점수 ────────────────────────────────────
        line1 = f"#{i}  ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm"
        line2 = f"ICP fit: {icp_fitness:.3f}"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness  = 1
        line_gap   = 4

        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        box_w = max(w1, w2) + 8
        box_h = h1 + h2 + line_gap + 8

        # 텍스트 박스 위치: bbox 왼쪽 상단 위에 배치, 화면 밖 나가면 아래로
        tx = max(x1, 0)
        ty = y1 - box_h - 4
        if ty < 0:
            ty = y2 + 4  # bbox 아래에 배치
        ty = min(ty, H - box_h - 2)
        tx = min(tx, W - box_w - 2)

        # 배경 박스 (반투명 효과: 검정 사각형)
        cv2.rectangle(out,
                      (tx - 2, ty),
                      (tx + box_w, ty + box_h),
                      (0, 0, 0), -1)

        # 텍스트 출력
        cv2.putText(out, line1, (tx + 2, ty + h1 + 2),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(out, line2, (tx + 2, ty + h1 + h2 + line_gap + 4),
                    font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    return out


def mask_nms(results, iou_threshold: float = MASK_IOU_THRESHOLD):
    """마스크 IoU 기반 NMS.

    두 인스턴스 마스크가 iou_threshold 이상 겹치면 score가 낮은 쪽 제거.
    results는 score 내림차순으로 정렬되어 있다고 가정 (RTMDetInferencer 기본 동작).

    Returns:
        keep: 살아남은 DetectionResult 리스트
        removed: 제거된 (제거된결과, 이긴결과) 튜플 리스트 (로그용)
    """
    keep    = []
    removed = []
    suppressed = [False] * len(results)

    for i, ri in enumerate(results):
        if suppressed[i]:
            continue
        keep.append(ri)
        area_i = ri.mask.sum()
        if area_i == 0:
            continue

        for j in range(i + 1, len(results)):
            if suppressed[j]:
                continue
            rj     = results[j]
            inter  = (ri.mask & rj.mask).sum()
            if inter == 0:
                continue
            area_j = rj.mask.sum()
            union  = area_i + area_j - inter
            iou    = inter / union if union > 0 else 0.0

            if iou >= iou_threshold:
                suppressed[j] = True
                removed.append((rj, ri, float(iou)))

    return keep, removed


def save_instance_pcd(points, out_path, color):
    if points.size == 0:
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float64), (len(points), 1))
    )
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def save_colored_full_pcd(pcd_organized, valid_mask, results, out_path):
    all_pts = pcd_organized[valid_mask]
    if len(all_pts) == 0:
        return False
    colors = np.tile(_BG_COLOR, (len(all_pts), 1))
    H, W = valid_mask.shape
    lookup = np.full((H, W), -1, dtype=np.int32)
    vr, vc = np.where(valid_mask)
    lookup[vr, vc] = np.arange(len(vr))
    for i, r in enumerate(results):
        ir, ic = np.where(r.mask & valid_mask)
        if len(ir):
            colors[lookup[ir, ic]] = _PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def run_detection(frame_name, gray, pcd_organized, valid_mask, inferencer, result_dir):
    H, W = gray.shape
    bgr  = np.stack([gray, gray, gray], axis=-1)
    results = inferencer.infer(bgr)

    # 마스크 IoU 기반 NMS — 중복 검출 제거
    results, nms_removed = mask_nms(results)
    if nms_removed:
        for rem, winner, iou in nms_removed:
            print(f"  [NMS] score={rem.score:.2f} 제거 "
                  f"(IoU={iou:.2f}, winner score={winner.score:.2f})", flush=True)

    cv2.imwrite(str(result_dir / f"{frame_name}_overlay.png"),
                overlay_results(bgr, results, valid_mask))
    save_colored_full_pcd(pcd_organized, valid_mask, results,
                          result_dir / f"{frame_name}_colored.ply")

    instances_info = []
    instance_plys  = []

    for i, r in enumerate(results):
        combined = r.mask & valid_mask
        obj_pts  = pcd_organized[combined]

        if len(obj_pts) < MIN_POINTS_PER_INSTANCE:
            instances_info.append({
                "instance_id": i, "class": r.class_name,
                "score": float(r.score),
                "skipped": "점이 너무 적음", "num_points": int(len(obj_pts)),
            })
            continue

        color_rgb = tuple(_PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)].tolist())
        ply_path  = result_dir / f"{frame_name}_obj{i}.ply"
        ok        = save_instance_pcd(obj_pts, ply_path, color=color_rgb)
        center    = obj_pts.mean(axis=0)
        size      = obj_pts.max(axis=0) - obj_pts.min(axis=0)

        # bbox 중심 (픽포인트 2D 위치 표시용)
        cx_2d = float((r.bbox[0] + r.bbox[2]) / 2)
        cy_2d = float((r.bbox[1] + r.bbox[3]) / 2)
        instances_info.append({
            "instance_id": i, "class": r.class_name,
            "score": float(r.score),
            "num_points_3d": int(len(obj_pts)),
            "center_mm": center.tolist(), "size_mm": size.tolist(),
            "bbox_center_2d": [cx_2d, cy_2d],
        })
        if ok:
            instance_plys.append((ply_path, cx_2d, cy_2d, r.bbox))

    summary = {
        "frame": frame_name,
        "num_detected": len(results),
        "num_with_pcd": len(instance_plys),
        "instances":    instances_info,
    }

    # 배경 PCD 생성 (valid 전체에서 인스턴스 마스크 제외)
    instance_mask_union = np.zeros(valid_mask.shape, dtype=bool)
    for r in results:
        instance_mask_union |= (r.mask & valid_mask)
    bg_only_mask = valid_mask & ~instance_mask_union
    bg_pts = pcd_organized[bg_only_mask]
    bg_pcd = o3d.geometry.PointCloud()
    if len(bg_pts) > 0:
        bg_pcd.points = o3d.utility.Vector3dVector(bg_pts / 1000.0)
        bg_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.55, 0.55, 0.55], (len(bg_pts), 1)))

    return summary, instance_plys, bgr, bg_pcd  # bgr: overlay용, bg_pcd: 배경 회색


# =============================================================================
# ICP  (기존과 동일)
# =============================================================================

def _Rx(d): c,s=np.cos(np.radians(d)),np.sin(np.radians(d)); R=np.eye(3); R[1,1]=c; R[1,2]=-s; R[2,1]=s; R[2,2]=c; return R
def _Ry(d): c,s=np.cos(np.radians(d)),np.sin(np.radians(d)); R=np.eye(3); R[0,0]=c; R[0,2]=s; R[2,0]=-s; R[2,2]=c; return R
def _Rz(d): c,s=np.cos(np.radians(d)),np.sin(np.radians(d)); R=np.eye(3); R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c; return R


def load_cad_as_pcd(cad_path):
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    ext  = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    if ext.max() > 10.0:
        mesh.scale(1.0 / 1000.0, center=np.zeros(3))
    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R      = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix  = np.eye(4); T_fix[:3, :3] = R; T_fix[:3, 3] = center - R @ center
    mesh.transform(T_fix)
    return mesh.sample_points_poisson_disk(CAD_SAMPLE_POINTS)


def run_icp_multistage(src, tgt, T_init):
    T = T_init.copy()
    for stage in ICP_STAGES:
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, stage["max_dist"], T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage["max_iter"]),
        )
        T = np.asarray(res.transformation)
    final = o3d.pipelines.registration.evaluate_registration(
        src, tgt, ICP_STAGES[-1]["max_dist"], T)
    return T, float(final.fitness), float(final.inlier_rmse)


def correct_flipped_pose(T, src, tgt):
    if T[:3, :3][2, 2] >= 0:
        final = o3d.pipelines.registration.evaluate_registration(
            src, tgt, ICP_STAGES[-1]["max_dist"], T)
        return T, float(final.fitness), float(final.inlier_rmse), False
    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4); T_flip[:3, :3] = R_flip
    c = T[:3, 3]; T_flip[:3, 3] = c - R_flip @ c
    T_f, fit, rmse = run_icp_multistage(src, tgt, T_flip @ T)
    return T_f, fit, rmse, True


def transform_to_pose(T):
    xyz_mm = (T[:3, 3] * 1000.0).tolist()
    R = T[:3, :3]
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2,1]/cp, R[2,2]/cp)
        yaw  = np.arctan2(R[1,0]/cp, R[0,0]/cp)
    else:
        roll, yaw = 0.0, np.arctan2(-R[0,1], R[1,1])
    e = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm": [round(v, 3) for v in xyz_mm],
        "euler_deg": {"roll_deg": round(e[0],4), "pitch_deg": round(e[1],4), "yaw_deg": round(e[2],4)},
        "transform_matrix": T.tolist(),
    }


def compute_pick_point(T):
    pl = CAD_PICK_LOCAL.copy()
    pl[0] += PICK_OFFSET_X_MM / 1000.0
    pl[1] += PICK_OFFSET_Y_MM / 1000.0
    pl[2] += PICK_OFFSET_Z_MM / 1000.0
    wt  = T @ pl
    pos = (wt[:3] * 1000.0).tolist()
    app = T[:3, 2] / (np.linalg.norm(T[:3, 2]) + 1e-9)
    R   = T[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R[2,0], np.sqrt(R[0,0]**2+R[1,0]**2))))
    cp = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = float(np.degrees(np.arctan2(R[2,1]/cp, R[2,2]/cp)))
        yaw  = float(np.degrees(np.arctan2(R[1,0]/cp, R[0,0]/cp)))
    else:
        roll, yaw = 0.0, float(np.degrees(np.arctan2(-R[0,1], R[1,1])))
    return {
        "position_mm":  [round(v, 3) for v in pos],
        "approach_deg": {"roll_deg": round(roll,4), "pitch_deg": round(pitch,4), "yaw_deg": round(yaw,4)},
    }


def build_icp_elements(scene_pcd, cad_pcd, T, pick, inst_color):
    """인스턴스 1개의 ICP 시각화 요소를 PCD로 반환 (누적용).
    inst_color: [R, G, B] 0~1 범위, 인스턴스별 구분색
    """
    # scene 포인트 — 인스턴스 고유색
    sv = copy.deepcopy(scene_pcd)
    sv.colors = o3d.utility.Vector3dVector(
        np.tile(inst_color, (len(np.asarray(sv.points)), 1)))

    # CAD — 초록 계열 (ICP 정합 결과)
    cv = copy.deepcopy(cad_pcd); cv.transform(T)
    cv.colors = o3d.utility.Vector3dVector(
        np.tile([0.1, 0.9, 0.3], (len(np.asarray(cv.points)), 1)))

    # 픽포인트 — 빨간 구
    pm  = np.array(pick["position_mm"]) / 1000.0
    sp  = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sp.translate(pm); sp.paint_uniform_color([1.0, 0.1, 0.1])
    sp_pcd = sp.sample_points_uniformly(500)

    # 접근 방향 벡터 — 파란 선 (approach_deg에서 Z축 방향 계산)
    deg   = pick["approach_deg"]
    cr, sr = np.cos(np.radians(deg["roll_deg"])),  np.sin(np.radians(deg["roll_deg"]))
    cp, sp = np.cos(np.radians(deg["pitch_deg"])), np.sin(np.radians(deg["pitch_deg"]))
    cy, sy = np.cos(np.radians(deg["yaw_deg"])),   np.sin(np.radians(deg["yaw_deg"]))
    app = np.array([cr*sy*sp + sr*cy, sr*sy - cr*cy*sp, cr*cp])
    app = app / (np.linalg.norm(app) + 1e-9)
    ap  = np.array([pm + t * app * 0.03 for t in np.linspace(0, 1, 50)])
    ap_pcd = o3d.geometry.PointCloud()
    ap_pcd.points = o3d.utility.Vector3dVector(ap)
    ap_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.3, 1.0], (50, 1)))

    return sv + cv + sp_pcd + ap_pcd


def run_icp_for_frame(instance_plys, cad_pcd, cad_down, result_dir, frame_name, bgr_image, bg_pcd=None):
    """
    모든 인스턴스 ICP 처리 후:
      - {frame_name}_colored.ply  : scene(인스턴스별색) + CAD + 픽포인트 + 접근벡터 통합
      - {frame_name}_result.json  : 모든 인스턴스 결과 통합
      - {frame_name}_overlay.png  : 픽포인트 좌표 텍스트 추가된 최종 이미지
    """
    icp_results  = []
    picks_2d     = []   # overlay 텍스트용 (px, py, pick)

    # 배경 회색 포인트로 combined_pcd 초기화
    if bg_pcd is not None:
        combined_pcd = bg_pcd
    else:
        combined_pcd = o3d.geometry.PointCloud()

    for ply_path, cx_2d, cy_2d, bbox in instance_plys:
        stem      = ply_path.stem
        inst_idx  = int(stem.split("obj")[-1])
        scene_pcd = o3d.io.read_point_cloud(str(ply_path))
        n_pts     = len(np.asarray(scene_pcd.points))
        if n_pts < 50:
            icp_results.append({"instance_id": inst_idx, "error": f"포인트 부족: {n_pts}개"})
            continue

        log(f"  obj{inst_idx}: {n_pts} pts")
        sc, _   = scene_pcd.remove_statistical_outlier(OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO)
        n_after = len(np.asarray(sc.points))
        sd      = sc.voxel_down_sample(VOXEL_SIZE_SCENE)

        T_init = np.eye(4)
        T_init[:3, 3] = np.asarray(sd.get_center()) - np.asarray(cad_down.get_center())

        T, fit, rmse = run_icp_multistage(cad_down, sd, T_init)
        T, fit, rmse, flipped = correct_flipped_pose(T, cad_down, sd)

        if flipped:
            log(f"    △ 뒤집힘 보정 후 fitness={fit:.4f}")

        if fit < ICP_FITNESS_THRESHOLD:
            log(f"    ✗ ICP 실패 (fitness={fit:.4f})")
            icp_results.append({"instance_id": inst_idx, "error": "ICP 정합 실패",
                                 "icp_fitness": float(fit)})
            # 인스턴스 PLY 삭제 (중간 파일 정리)
            ply_path.unlink(missing_ok=True)
            continue

        if max(abs(v) for v in T[:3, 3]) > XYZ_MAX_M:
            icp_results.append({"instance_id": inst_idx, "error": "xyz 범위 이상",
                                 "icp_fitness": float(fit)})
            ply_path.unlink(missing_ok=True)
            continue

        pose = transform_to_pose(T)
        pick = compute_pick_point(T)
        ppos = pick["position_mm"]
        deg  = pick["approach_deg"]

        # 인스턴스 고유색 (PALETTE RGB)
        inst_color = _PALETTE_RGB_FLOAT[inst_idx % len(_PALETTE_RGB_FLOAT)].tolist()

        # ICP 시각화 요소를 통합 PCD에 누적
        combined_pcd += build_icp_elements(scene_pcd, cad_pcd, T, pick, inst_color)

        # overlay용 2D 위치 기록 (cx_2d, cy_2d, pick, icp_fitness, bbox)
        picks_2d.append((cx_2d, cy_2d, pick, float(fit), bbox))

        result = {
            "instance_id":   inst_idx,
            "icp_fitness":   float(fit),
            "icp_rmse_m":    float(rmse),
            "was_flipped":   flipped,
            "num_points_scene": n_pts,
            "num_points_after_outlier_removal": n_after,
            "pose":          pose,
            "pick_point":    pick,
        }
        print(f"    ✓ 픽포인트: ({ppos[0]:.1f}, {ppos[1]:.1f}, {ppos[2]:.1f}) mm  "
              f"fit={fit:.3f}  roll={deg['roll_deg']:.2f}  "
              f"pitch={deg['pitch_deg']:.2f}  yaw={deg['yaw_deg']:.2f}", flush=True)
        icp_results.append(result)

        # 중간 인스턴스 PLY 삭제 (통합 PLY로 대체)
        ply_path.unlink(missing_ok=True)

    # ── 통합 PLY 저장 (frame_colored.ply 덮어쓰기) ──────────────────────────
    if len(np.asarray(combined_pcd.points)) > 0:
        ply_out = result_dir / f"{frame_name}_colored.ply"
        o3d.io.write_point_cloud(str(ply_out), combined_pcd, write_ascii=False)
        log(f"  ✓ 통합 PLY: {ply_out.name}")

    # ── 통합 JSON 저장 ────────────────────────────────────────────────────────
    success = [r for r in icp_results if "error" not in r]
    json_out = result_dir / f"{frame_name}_result.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump({
            "frame":      frame_name,
            "num_total":  len(icp_results),
            "num_success": len(success),
            "instances":  icp_results,
        }, f, indent=2, ensure_ascii=False)
    log(f"  ✓ 통합 JSON: {json_out.name}")

    # ── overlay PNG에 픽포인트 좌표 추가 ─────────────────────────────────────
    if picks_2d:
        overlay_final = draw_picks_on_overlay(bgr_image, picks_2d)
        overlay_out   = result_dir / f"{frame_name}_overlay.png"
        cv2.imwrite(str(overlay_out), overlay_final)
        log(f"  ✓ overlay PNG: {overlay_out.name}")

    return icp_results


# =============================================================================
# TCP 통신 헬퍼
# =============================================================================

def send_response(conn: socket.socket, payload: dict) -> None:
    """payload를 로봇 파싱용 포맷으로 직렬화 후 전송.

    성공:  {'ok', N, (x, y, z, roll, pitch, yaw, fit), ...}\n
    미검출: {'No'}\n
    오류:  {'error', 'message'}\n
    """
    conn.sendall((format_response(payload) + "\n").encode("utf-8"))


def format_response(payload: dict) -> str:
    """payload dict → 로봇 파싱용 문자열 변환.

    성공 예:
        {'ok', 3, (84.6, -4.1, 513.6, 1.69, 9.42, 1.36, 0.91), (...), (...)}
    미검출:
        {'No'}
    오류:
        {'error', 'ICP 정합 실패'}
    """
    status = payload.get("status")

    if status == "ok":
        picks  = payload["picks"]
        parts  = [f"'ok'", str(len(picks))]
        for pk in picks:
            pp  = pk["position_mm"]
            deg = pk["approach_deg"]
            fit = round(pk["icp_fitness"], 2)   # 소수점 셋째 자리에서 반올림 → 둘째까지
            tup = (
                round(pp[0], 3),
                round(pp[1], 3),
                round(pp[2], 3),
                round(deg["roll_deg"],  3),
                round(deg["pitch_deg"], 3),
                round(deg["yaw_deg"],   3),
                fit,
            )
            parts.append(str(tup))
        return "{" + ", ".join(parts) + "}"

    elif status in ("no_object", "No"):
        return "{'No'}"

    else:  # error
        msg = payload.get("message", "unknown error")
        return "{" + f"'error', '{msg}'" + "}"


# =============================================================================
# 한 프레임 처리 — 카메라 없는 버전
# =============================================================================

def process_one_frame_offline(
    intensity_path: Path,
    pcd_npy_path:   Path,
    mask_npy_path:  Path,
    frame_name:     str,
    out_dir:        Path,
    inferencer:     RTMDetInferencer,
    cad_pcd,
    cad_down,
) -> dict:
    """
    5_Run_binpicking_TCP.py 의 process_one_frame() 과 동일한 로직.
    cam.capture() 대신 load_frame_from_files() 로 데이터를 주입.
    반환값: format_response() 에 넘길 payload dict
    """
    _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "-" * 70
    print(f"\n{sep}", flush=True)
    print(f"  [{frame_name}] 오프라인 처리  {_now}", flush=True)

    result_dir = out_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 (cam.capture() 대체) ────────────────────────────────────
    gray, pcd_organized, valid_mask = load_frame_from_files(
        intensity_path, pcd_npy_path, mask_npy_path
    )

    # ── Detection ───────────────────────────────────────────────────────────
    print("  [Detection]", flush=True)
    t0 = time.perf_counter()
    summary, inst_plys, bgr_image, bg_pcd = run_detection(
        frame_name, gray, pcd_organized, valid_mask,
        inferencer, result_dir
    )
    det_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  검출: {summary['num_detected']}개  PCD: {summary['num_with_pcd']}개"
          f"  ({det_ms:.0f} ms)", flush=True)

    if not inst_plys:
        print("  브라켓 없음", flush=True)
        return {"status": "No"}

    # ── ICP ─────────────────────────────────────────────────────────────────
    print("  [ICP]", flush=True)
    t0 = time.perf_counter()
    icp_results = run_icp_for_frame(
        inst_plys, cad_pcd, cad_down, result_dir, frame_name, bgr_image,
        bg_pcd=bg_pcd,
    )
    icp_ms = (time.perf_counter() - t0) * 1000.0

    success = [r for r in icp_results if "error" not in r]
    n_fail  = len(icp_results) - len(success)
    print(f"  ICP: 성공 {len(success)}개  실패 {n_fail}개  ({icp_ms:.0f} ms)", flush=True)

    if not success:
        return {"status": "No"}

    picks = [
        {
            "position_mm":  r["pick_point"]["position_mm"],
            "approach_deg": r["pick_point"]["approach_deg"],
            "icp_fitness":  r["icp_fitness"],
        }
        for r in success
    ]

    for i, pk in enumerate(picks):
        pp  = pk["position_mm"]
        deg = pk["approach_deg"]
        fit = pk["icp_fitness"]
        print(f"  #{i}  위치: ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm  fit={fit:.2f}"
              f"  roll={deg['roll_deg']:.2f}  pitch={deg['pitch_deg']:.2f}"
              f"  yaw={deg['yaw_deg']:.2f}", flush=True)

    return {"status": "ok", "picks": picks}


# =============================================================================
# 메인
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="오프라인 빈피킹 파이프라인 (카메라 없음)")
    p.add_argument("--intensity",    type=Path, default=None,
                   help="intensity PNG 경로 (직접 지정 시)")
    p.add_argument("--pcd_npy",      type=Path, default=None,
                   help="pointcloud_organized .npy 경로 (직접 지정 시)")
    p.add_argument("--mask_npy",     type=Path, default=None,
                   help="valid_mask .npy 경로 (직접 지정 시)")
    p.add_argument("--input",        type=Path,
                   default=ROOT / "data" / "offline_input" ,
                   help="입력 디렉토리 (TCP 스크립트 --out 경로와 동일한 서브폴더 구조)")
    p.add_argument("--out",          type=Path,
                   default=ROOT / "data" / "offline_output",
                   help="결과 저장 루트 (results/ 하위에 생성됨)")
    p.add_argument("--interactive",  action="store_true",
                   help="파일을 직접 골라가며 반복 실행")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 사전 점검
    for path, name in [
        (CONFIG_PATH,     "RTMDet config"),
        (CHECKPOINT_PATH, "RTMDet checkpoint"),
        (CAD_PATH,        "CAD STL"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} 없음: {path}", flush=True)
            return 1

    print("=" * 70)
    print("  오프라인 빈피킹 파이프라인")
    print("=" * 70)
    print(f"  Checkpoint : {CHECKPOINT_PATH.name}")
    print(f"  CAD        : {CAD_PATH.name}")
    print(f"  입력       : {args.input}")
    print(f"    intensity/            pointcloud_organized/  valid_mask/")
    print(f"  결과       : {args.out / 'results'}")
    print("=" * 70)

    # ── [1] 모델 로드 ─────────────────────────────────────────────────────────
    print("\n[1] RTMDet 모델 로드 중...", flush=True)
    inferencer = RTMDetInferencer(
        config=CONFIG_PATH,
        checkpoint=CHECKPOINT_PATH,
        device="cuda:0",
        score_threshold=SCORE_THRESHOLD,
    )
    print(f"    클래스: {inferencer.class_names}", flush=True)

    # ── [2] CAD 로드 ──────────────────────────────────────────────────────────
    print("\n[2] CAD 모델 로드 중...", flush=True)
    try:
        cad_pcd = load_cad_as_pcd(CAD_PATH)
    except Exception as e:
        print(f"ERROR: CAD 로드 실패: {e}", flush=True)
        return 1
    cad_down = cad_pcd.voxel_down_sample(VOXEL_SIZE_CAD)
    print(f"    {len(np.asarray(cad_pcd.points))}pts  "
          f"다운샘플: {len(np.asarray(cad_down.points))}pts", flush=True)

    # ── [3] 파일 루프 ─────────────────────────────────────────────────────────
    frame_idx = 0

    while True:
        frame_idx += 1
        frame_name = f"frame_{frame_idx:04d}"

        try:
            # 파일 경로 결정 (세 가지 모드)
            if args.intensity and args.pcd_npy and args.mask_npy:
                # 모드 A: CLI로 직접 지정
                intensity_path = args.intensity
                pcd_npy_path   = args.pcd_npy
                mask_npy_path  = args.mask_npy
                loop = False

            elif args.interactive:
                # 모드 B: 대화형 — 목록에서 선택
                intensity_path, pcd_npy_path, mask_npy_path = pick_set_interactive(args.input)
                loop = True

            else:
                # 모드 C: 자동 — 최신 파일 세트
                print(f"\n[자동탐색] {args.input} 에서 최신 파일 세트...", flush=True)
                intensity_path, pcd_npy_path, mask_npy_path = find_latest_set(args.input)
                loop = False

            payload = process_one_frame_offline(
                intensity_path, pcd_npy_path, mask_npy_path,
                frame_name, args.out,
                inferencer, cad_pcd, cad_down,
            )

        except KeyboardInterrupt:
            print("\n중단됨.", flush=True)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            payload = {"status": "error", "message": str(e)}

        # TCP 응답 포맷과 동일하게 출력
        msg = format_response(payload)
        print(f"\n[응답 미리보기]\n  {msg}", flush=True)

        if not loop:
            break

        # 대화형: 계속 여부 확인
        try:
            ans = input("\n다음 프레임 처리? [y/n] (기본=y): ").strip().lower()
        except EOFError:
            break
        if ans == "n":
            break

    print(f"\n완료. 결과: {args.out / 'results'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())