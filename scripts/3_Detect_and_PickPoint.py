"""빈피킹 파이프라인: 2D Detection → PCD 분리 → ICP 정합 → 픽포인트 산출.

Stage 5-A + Stage 5-B 통합본.

파이프라인:
    [A] RTMDet-Ins로 intensity 이미지에서 브라켓 인스턴스 검출
    [A] Detection 마스크 → organized PCD 매핑 → 인스턴스별 PLY 저장
    [B] CAD STL ↔ 인스턴스 PCD ICP 정합 → 6DoF 자세 추정
    [B] 픽포인트(위치 + 접근방향) 계산 + 시각화 PLY 저장

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/5_binpicking_pipeline.py

입력:
    data/dataset_input/{input_data}/intensity/          ← 강도 이미지 (frame_*.png)
    data/dataset_input/{input_data}/pointcloud_organized/ ← organized PCD (.npy)
    data/dataset_input/{input_data}/valid_mask/          ← 유효 마스크 (.npy)
    data/cad/bracket_v2.stl                              ← CAD 모델 (mm 단위)

출력:
    data/inference_results/{input_data}/
    ├── frame_NNNN_overlay.png         ← [A] 2D detection 시각화
    ├── frame_NNNN_colored.ply         ← [A] 전체 PCD (배경 회색 + 인스턴스 컬러)
    ├── frame_NNNN_obj{i}.ply          ← [A] 인스턴스 단독 PCD
    ├── frame_NNNN_summary.json        ← [A] detection 통계
    ├── frame_NNNN_obj{i}_icp_vis.ply  ← [B] ICP 정합 시각화 (scene+CAD+픽포인트)
    ├── frame_NNNN_obj{i}_pose.json    ← [B] 6DoF 자세 + 픽포인트
    └── icp_summary.json               ← [B] ICP 전체 요약
"""

from __future__ import annotations

import copy
import json
import sys
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
# 공통 설정
# =============================================================================
input_data = "20260519_0000"

# -----------------------------------------------------------------------------
# [A] Detection 설정
# -----------------------------------------------------------------------------
WORK_DIR       = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1"
CONFIG_PATH    = WORK_DIR / "rtmdet-ins_bracket.py"
CHECKPOINT_PATH = WORK_DIR / "best_coco_bbox_mAP_epoch_50.pth"

DATASET_DIR       = ROOT / "data" / "dataset_input" / input_data
INTENSITY_DIR     = DATASET_DIR / "intensity"
PCD_ORGANIZED_DIR = DATASET_DIR / "pointcloud_organized"
VALID_MASK_DIR    = DATASET_DIR / "valid_mask"

OUTPUT_DIR = ROOT / "data" / "inference_results" / input_data

SCORE_THRESHOLD      = 0.3    # detection 신뢰도 임계값
MIN_POINTS_PER_INSTANCE = 100 # 이 이하 포인트 인스턴스는 노이즈로 제거

# 인스턴스 색상 팔레트 (2D overlay BGR / 3D PLY RGB 공유)
_PALETTE_BGR = np.array([
    [ 50,  50, 255],  # 빨강
    [ 50, 200,  50],  # 초록
    [255, 100,  50],  # 청록
    [ 30, 180, 255],  # 주황
    [230,  50, 180],  # 자홍
    [200, 200,  30],  # 노랑
], dtype=np.uint8)
_PALETTE_RGB_FLOAT = _PALETTE_BGR[:, ::-1].astype(np.float64) / 255.0
_BG_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float64)  # 배경 회색

# -----------------------------------------------------------------------------
# [B] ICP 설정
# -----------------------------------------------------------------------------
CAD_PATH = ROOT / "data" / "cad" / "bracket_v2.stl"

CAD_SAMPLE_POINTS = 20000

VOXEL_SIZE_CAD   = 0.002   # 2mm
VOXEL_SIZE_SCENE = 0.003   # 3mm

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

ICP_STAGES = [
    {"max_dist": 0.020, "max_iter": 100},   # 20mm — 초기 오차 흡수
    {"max_dist": 0.010, "max_iter": 100},   # 10mm — 중간 수렴
    {"max_dist": 0.005, "max_iter": 100},   # 5mm  — 정밀 수렴
]
ICP_FITNESS_THRESHOLD = 0.5

XYZ_MAX_M = 2.0  # 센서 범위 초과 시 로컬 미니멈으로 판정

# CAD 축 보정 (Rx=-90, Ry=90, Rz=90 — 시각적으로 확인된 값)
CAD_AXIS_CORRECTION_DEG = (-90, 90, 90)

# 픽포인트 — 축 보정 후 CAD 로컬 좌표계에서 상단 수평면 중심 실측값 (단위: m)
CAD_PICK_LOCAL = np.array([0.000, -0.100, 0.031, 1.0])

# 픽포인트 미세 조정 오프셋 (CAD 로컬 좌표계 기준, 단위: mm)
# X: 브라켓 폭 방향   (+: 오른쪽, -: 왼쪽)
# Y: 브라켓 길이 방향  (+: 앞,    -: 뒤)
# Z: 브라켓 높이 방향  (+: 위,    -: 아래)
PICK_OFFSET_X_MM = -5.0   # mm
PICK_OFFSET_Y_MM =  0.0   # mm
PICK_OFFSET_Z_MM =  0.0   # mm


# =============================================================================
# [A] Detection + PCD 분리
# =============================================================================

def overlay_results(image_bgr: np.ndarray, results, valid_mask=None) -> np.ndarray:
    """Detection 결과 + valid_mask를 2D 이미지에 시각화."""
    overlay = image_bgr.copy()
    if valid_mask is not None:
        overlay[~valid_mask] = (overlay[~valid_mask] * 0.4).astype(np.uint8)
    for i, r in enumerate(results):
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        color_layer = np.zeros_like(overlay)
        color_layer[r.mask] = color
        overlay[r.mask] = (0.5 * overlay[r.mask] + 0.5 * color_layer[r.mask]).astype(np.uint8)
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


def crop_pcd_by_mask(
    pcd_organized: np.ndarray,
    valid_mask: np.ndarray,
    instance_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detection 마스크 & valid_mask 교집합으로 객체 PCD 추출."""
    combined = instance_mask & valid_mask
    return pcd_organized[combined], combined


def save_instance_pcd(points: np.ndarray, out_path: Path, color: tuple) -> bool:
    """인스턴스 단독 PCD를 PLY로 저장 (mm → m)."""
    if points.size == 0:
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float64), (len(points), 1))
    )
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def save_colored_full_pcd(
    pcd_organized: np.ndarray,
    valid_mask: np.ndarray,
    results,
    out_path: Path,
) -> bool:
    """전체 PCD를 단일 PLY로 저장 (배경 회색, 인스턴스별 컬러)."""
    all_points = pcd_organized[valid_mask]
    if len(all_points) == 0:
        return False

    colors = np.tile(_BG_COLOR, (len(all_points), 1))

    H, W = valid_mask.shape
    lookup = np.full((H, W), -1, dtype=np.int32)
    vr, vc = np.where(valid_mask)
    lookup[vr, vc] = np.arange(len(vr))

    for i, r in enumerate(results):
        ir, ic = np.where(r.mask & valid_mask)
        if len(ir) == 0:
            continue
        colors[lookup[ir, ic]] = _PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def process_frame_detection(
    frame_name: str,
    inferencer: RTMDetInferencer,
    output_dir: Path,
) -> dict:
    """한 프레임의 Detection + PCD 분리 처리 (Stage 5-A)."""

    intensity_path  = INTENSITY_DIR / f"{frame_name}.png"
    pcd_path        = PCD_ORGANIZED_DIR / f"{frame_name}.npy"
    valid_mask_path = VALID_MASK_DIR / f"{frame_name}.npy"

    if not all(p.exists() for p in [intensity_path, pcd_path, valid_mask_path]):
        return {"error": "파일 없음", "frame": frame_name}

    gray         = cv2.imread(str(intensity_path), cv2.IMREAD_GRAYSCALE)
    bgr          = np.stack([gray, gray, gray], axis=-1)
    pcd_organized = np.load(pcd_path)    # (H, W, 3) float32, NaN=invalid
    valid_mask   = np.load(valid_mask_path)  # (H, W) bool
    H, W         = gray.shape

    # Detection
    results = inferencer.infer(bgr)

    # 2D overlay 저장
    overlay = overlay_results(bgr, results, valid_mask)
    cv2.imwrite(str(output_dir / f"{frame_name}_overlay.png"), overlay)

    # 전체 colored PCD 저장
    colored_ply_path = output_dir / f"{frame_name}_colored.ply"
    colored_ok = save_colored_full_pcd(pcd_organized, valid_mask, results, colored_ply_path)
    if colored_ok:
        print(f"    colored PLY: {colored_ply_path.name}", flush=True)

    # 인스턴스별 PCD 분리 저장
    instances_info = []
    instance_plys  = []  # ICP 단계로 넘길 경로 목록

    for i, r in enumerate(results):
        object_points, combined_mask = crop_pcd_by_mask(pcd_organized, valid_mask, r.mask)

        if len(object_points) < MIN_POINTS_PER_INSTANCE:
            instances_info.append({
                "instance_id": i, "class": r.class_name,
                "score": float(r.score),
                "skipped": "점이 너무 적음", "num_points": int(len(object_points)),
            })
            continue

        color_rgb = tuple(_PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)].tolist())
        ply_path  = output_dir / f"{frame_name}_obj{i}.ply"
        ok        = save_instance_pcd(object_points, ply_path, color=color_rgb)

        center    = object_points.mean(axis=0)
        pcd_range = object_points.max(axis=0) - object_points.min(axis=0)

        info = {
            "instance_id": i,
            "class": r.class_name,
            "score": float(r.score),
            "num_pixels_detection": int(r.mask.sum()),
            "num_pixels_after_valid": int(combined_mask.sum()),
            "num_points_3d": int(len(object_points)),
            "valid_overlap_ratio": float(combined_mask.sum()) / max(int(r.mask.sum()), 1),
            "bbox_2d": r.bbox.tolist(),
            "center_mm": center.tolist(),
            "size_mm": pcd_range.tolist(),
            "z_median_mm": float(np.median(object_points[:, 2])),
            "ply_saved": ok,
            "ply_path": str(ply_path.relative_to(ROOT)) if ok else None,
        }
        instances_info.append(info)
        if ok:
            instance_plys.append(ply_path)

    summary = {
        "frame": frame_name,
        "image_size": [int(H), int(W)],
        "valid_mask_ratio": float(valid_mask.mean()),
        "num_detected": len(results),
        "num_with_pcd": len([x for x in instances_info if "skipped" not in x]),
        "colored_ply_saved": colored_ok,
        "colored_ply_path": str(colored_ply_path.relative_to(ROOT)) if colored_ok else None,
        "instances": instances_info,
    }
    with (output_dir / f"{frame_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, instance_plys


# =============================================================================
# [B] ICP 정합 + 픽포인트
# =============================================================================

def _Rx(d: float) -> np.ndarray:
    a = np.radians(d)
    return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])

def _Ry(d: float) -> np.ndarray:
    a = np.radians(d)
    return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])

def _Rz(d: float) -> np.ndarray:
    a = np.radians(d)
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])


def load_cad_as_pcd(stl_path: Path) -> o3d.geometry.PointCloud:
    """STL 로드 → mm→m 변환 → 축 보정 → 포인트 샘플링."""
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if not mesh.has_triangles():
        raise ValueError(f"STL 로드 실패: {stl_path}")

    extent_before = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    STL 원본 extent: {np.round(extent_before, 2)} mm", flush=True)
    mesh.scale(1.0 / 1000.0, center=np.zeros(3))  # 원점 기준 스케일 (버그 방지)
    extent_after = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    변환 후 extent:  {np.round(extent_after, 4)} m  "
          f"center={np.round(np.asarray(mesh.get_center()), 4)}", flush=True)

    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix = np.eye(4)
    T_fix[:3, :3] = R
    T_fix[:3, 3] = center - R @ center
    mesh.transform(T_fix)
    print(f"    축 보정: Rx={rx}° Ry={ry}° Rz={rz}°", flush=True)
    print(f"    CAD 픽포인트 로컬: "
          f"X={CAD_PICK_LOCAL[0]*1000:.1f}mm "
          f"Y={CAD_PICK_LOCAL[1]*1000:.1f}mm "
          f"Z={CAD_PICK_LOCAL[2]*1000:.1f}mm", flush=True)

    return mesh.sample_points_poisson_disk(CAD_SAMPLE_POINTS)


def run_icp_multistage(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """coarse→fine 다단계 Point-to-Point ICP."""
    T = init_transform.copy()
    for stage in ICP_STAGES:
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            stage["max_dist"], T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage["max_iter"]),
        )
        T = np.asarray(result.transformation)
    final = o3d.pipelines.registration.evaluate_registration(
        source, target, ICP_STAGES[-1]["max_dist"], T
    )
    return T, float(final.fitness), float(final.inlier_rmse)


def correct_flipped_pose(
    T: np.ndarray,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, float, float, bool]:
    """R[2,2] < 0 이면 Z축 180도 회전 후 ICP 재수렴."""
    R = T[:3, :3]
    if R[2, 2] >= 0:
        final = o3d.pipelines.registration.evaluate_registration(
            source, target, ICP_STAGES[-1]["max_dist"], T
        )
        return T, float(final.fitness), float(final.inlier_rmse), False

    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4)
    T_flip[:3, :3] = R_flip
    c = T[:3, 3]
    T_flip[:3, 3] = c - R_flip @ c
    T_final, fitness, rmse = run_icp_multistage(source, target, T_flip @ T)
    return T_final, fitness, rmse, True


def transform_to_pose(T: np.ndarray) -> dict:
    """4x4 변환행렬(m) → xyz mm + ZYX 오일러각 deg."""
    xyz_mm = (T[:3, 3] * 1000.0).tolist()
    R = T[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    euler = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm": [round(v, 3) for v in xyz_mm],
        "euler_deg": {
            "roll_deg":  round(euler[0], 4),
            "pitch_deg": round(euler[1], 4),
            "yaw_deg":   round(euler[2], 4),
        },
        "transform_matrix": T.tolist(),
    }


def compute_pick_point(T: np.ndarray) -> dict:
    """ICP 변환행렬로부터 픽포인트 위치 + 접근 방향 계산."""
    pick_local = CAD_PICK_LOCAL.copy()
    pick_local[0] += PICK_OFFSET_X_MM / 1000.0
    pick_local[1] += PICK_OFFSET_Y_MM / 1000.0
    pick_local[2] += PICK_OFFSET_Z_MM / 1000.0

    world_top    = T @ pick_local
    pick_pos_mm  = (world_top[:3] * 1000.0).tolist()

    approach = T[:3, 2] / (np.linalg.norm(T[:3, 2]) + 1e-9)

    R = T[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))))
    cp = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = float(np.degrees(np.arctan2(R[2,1]/cp, R[2,2]/cp)))
        yaw  = float(np.degrees(np.arctan2(R[1,0]/cp, R[0,0]/cp)))
    else:
        roll, yaw = 0.0, float(np.degrees(np.arctan2(-R[0,1], R[1,1])))

    return {
        "position_mm":  [round(v, 3) for v in pick_pos_mm],
        "approach_vec": [round(v, 6) for v in approach.tolist()],
        "approach_deg": {
            "roll_deg":  round(roll,  4),
            "pitch_deg": round(pitch, 4),
            "yaw_deg":   round(yaw,   4),
        },
    }


def save_icp_visualization(
    scene_pcd: o3d.geometry.PointCloud,
    cad_pcd: o3d.geometry.PointCloud,
    T: np.ndarray,
    pick: dict,
    out_path: Path,
) -> None:
    """scene(회색) + CAD(초록) + 픽포인트(빨강 구) + 접근방향(파랑)을 PLY로 저장."""
    scene_vis = copy.deepcopy(scene_pcd)
    n = len(np.asarray(scene_vis.points))
    scene_vis.colors = o3d.utility.Vector3dVector(np.tile([0.6, 0.6, 0.6], (n, 1)))

    cad_vis = copy.deepcopy(cad_pcd)
    cad_vis.transform(T)
    n = len(np.asarray(cad_vis.points))
    cad_vis.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.9, 0.3], (n, 1)))

    pick_pos_m = np.array(pick["position_mm"]) / 1000.0
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.translate(pick_pos_m)
    sphere.paint_uniform_color([1.0, 0.1, 0.1])
    sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)

    approach  = np.array(pick["approach_vec"])
    arrow_pts = np.array([pick_pos_m + t * approach * 0.03 for t in np.linspace(0, 1, 50)])
    arrow_pcd = o3d.geometry.PointCloud()
    arrow_pcd.points = o3d.utility.Vector3dVector(arrow_pts)
    arrow_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.3, 1.0], (50, 1)))

    o3d.io.write_point_cloud(str(out_path), scene_vis + cad_vis + sphere_pcd + arrow_pcd,
                             write_ascii=False)


def process_instance_icp(
    instance_ply: Path,
    cad_pcd: o3d.geometry.PointCloud,
    cad_down: o3d.geometry.PointCloud,
    output_dir: Path,
) -> dict:
    """인스턴스 PLY 1개에 대해 중심정렬 + 다단계 ICP + 픽포인트 계산 (Stage 5-B)."""
    stem = instance_ply.stem

    scene_pcd = o3d.io.read_point_cloud(str(instance_ply))
    n_pts = len(np.asarray(scene_pcd.points))
    if n_pts < 50:
        return {"file": stem, "error": f"포인트 부족: {n_pts}개"}

    print(f"  {stem}: {n_pts} pts", flush=True)

    # 전처리 (노이즈 제거 + 다운샘플)
    scene_clean, _ = scene_pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO
    )
    n_after      = len(np.asarray(scene_clean.points))
    scene_down   = scene_clean.voxel_down_sample(VOXEL_SIZE_SCENE)
    removal_pct  = (1 - n_after / max(n_pts, 1)) * 100
    print(f"    노이즈 제거: {n_pts} → {n_after} pts ({removal_pct:.1f}% 제거)", flush=True)
    print(f"    다운샘플:    scene={len(np.asarray(scene_down.points))}  "
          f"cad={len(np.asarray(cad_down.points))}", flush=True)

    # 중심 정렬 초기화
    src_c  = np.asarray(cad_down.get_center())
    tgt_c  = np.asarray(scene_down.get_center())
    T_init = np.eye(4)
    T_init[:3, 3] = tgt_c - src_c
    print(f"    중심 정렬:   {np.round(src_c,3)} → {np.round(tgt_c,3)}", flush=True)

    # 다단계 ICP
    print(f"    ICP 중...", flush=True)
    T_final, fitness, rmse = run_icp_multistage(cad_down, scene_down, T_init)
    print(f"    ICP fitness={fitness:.4f}, rmse={rmse:.6f}", flush=True)

    # 뒤집힘 보정
    T_final, fitness, rmse, was_flipped = correct_flipped_pose(T_final, cad_down, scene_down)
    if was_flipped:
        print(f"    △ 뒤집힘 감지 → Z축 180도 보정 후 재수렴", flush=True)
        print(f"    보정 후 fitness={fitness:.4f}, rmse={rmse:.6f}", flush=True)

    t_mm = np.round(T_final[:3, 3] * 1000, 1)
    print(f"    T translation: {t_mm} mm", flush=True)

    if fitness < ICP_FITNESS_THRESHOLD:
        print(f"    ✗ ICP 실패 (fitness={fitness:.4f} < {ICP_FITNESS_THRESHOLD})", flush=True)
        return {"file": stem, "error": "ICP 정합 실패",
                "icp_fitness": float(fitness), "icp_rmse": float(rmse)}

    if max(abs(v) for v in T_final[:3, 3]) > XYZ_MAX_M:
        print(f"    ✗ xyz 비정상: {t_mm} mm → 로컬 미니멈", flush=True)
        return {"file": stem, "error": "xyz 범위 이상", "icp_fitness": float(fitness)}

    pose = transform_to_pose(T_final)
    pick = compute_pick_point(T_final)
    xyz  = pose["xyz_mm"]
    eul  = pose["euler_deg"]
    ppos = pick["position_mm"]
    avec = pick["approach_vec"]
    adeg = pick["approach_deg"]

    vis_path = output_dir / f"{stem}_icp_vis.ply"
    save_icp_visualization(scene_pcd, cad_pcd, T_final, pick, vis_path)

    result = {
        "file": stem,
        "input_ply": str(instance_ply.relative_to(ROOT)),
        "num_points_scene": n_pts,
        "num_points_after_outlier_removal": n_after,
        "icp_fitness": float(fitness),
        "icp_rmse_m": float(rmse),
        "was_flipped": was_flipped,
        "pose": pose,
        "pick_point": pick,
        "vis_ply": str(vis_path.relative_to(ROOT)),
    }
    with (output_dir / f"{stem}_pose.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"    ✓ CAD 중심:  xyz=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm  "
          f"roll={eul['roll_deg']:.1f}° pitch={eul['pitch_deg']:.1f}° "
          f"yaw={eul['yaw_deg']:.1f}°", flush=True)
    print(f"    ✓ 픽포인트:  xyz=({ppos[0]:.1f}, {ppos[1]:.1f}, {ppos[2]:.1f}) mm", flush=True)
    print(f"    ✓ 접근 방향: vec=({avec[0]:.3f}, {avec[1]:.3f}, {avec[2]:.3f})  "
          f"roll={adeg['roll_deg']:.1f}° pitch={adeg['pitch_deg']:.1f}° "
          f"yaw={adeg['yaw_deg']:.1f}°", flush=True)

    return result


# =============================================================================
# 메인
# =============================================================================

def main() -> int:
    # 사전 점검
    for path, name in [
        (CONFIG_PATH,       "RTMDet config"),
        (CHECKPOINT_PATH,   "RTMDet checkpoint"),
        (INTENSITY_DIR,     "intensity dir"),
        (PCD_ORGANIZED_DIR, "organized PCD dir"),
        (VALID_MASK_DIR,    "valid mask dir"),
        (CAD_PATH,          "CAD STL"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} 없음: {path}", flush=True)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 헤더 출력 ────────────────────────────────────────────────────────────
    print("=" * 70, flush=True)
    print("  빈피킹 파이프라인: Detection + PCD 분리 + ICP 정합", flush=True)
    print("=" * 70, flush=True)
    print(f"  Dataset:     {DATASET_DIR}", flush=True)
    print(f"  Output:      {OUTPUT_DIR}", flush=True)
    print(f"  Score thr:   {SCORE_THRESHOLD}", flush=True)
    print(f"  CAD:         {CAD_PATH.name}", flush=True)
    print(f"  축 보정:     Rx={CAD_AXIS_CORRECTION_DEG[0]}° "
          f"Ry={CAD_AXIS_CORRECTION_DEG[1]}° Rz={CAD_AXIS_CORRECTION_DEG[2]}°", flush=True)
    stages_str = " → ".join(f"{s['max_dist']*1000:.0f}mm×{s['max_iter']}" for s in ICP_STAGES)
    print(f"  ICP 단계:    {stages_str}", flush=True)
    print(f"  픽 오프셋:   X={PICK_OFFSET_X_MM:+.1f}mm  "
          f"Y={PICK_OFFSET_Y_MM:+.1f}mm  Z={PICK_OFFSET_Z_MM:+.1f}mm", flush=True)
    print("=" * 70, flush=True)

    # ── [1] RTMDet 모델 로드 ─────────────────────────────────────────────────
    print("\n[1] RTMDet 모델 로드 중...", flush=True)
    inferencer = RTMDetInferencer(
        config=CONFIG_PATH,
        checkpoint=CHECKPOINT_PATH,
        device="cuda:0",
        score_threshold=SCORE_THRESHOLD,
    )
    print(f"    ✓ 클래스: {inferencer.class_names}", flush=True)

    # ── [2] CAD 로드 ─────────────────────────────────────────────────────────
    print("\n[2] CAD 모델 로드 중...", flush=True)
    try:
        cad_pcd = load_cad_as_pcd(CAD_PATH)
    except Exception as e:
        print(f"ERROR: CAD 로드 실패: {e}", flush=True)
        return 1
    cad_down = cad_pcd.voxel_down_sample(VOXEL_SIZE_CAD)
    print(f"    ✓ CAD 샘플: {len(np.asarray(cad_pcd.points))}pts  "
          f"다운샘플: {len(np.asarray(cad_down.points))}pts "
          f"(voxel={VOXEL_SIZE_CAD*1000:.1f}mm)", flush=True)

    # ── [3] 프레임별 처리 ────────────────────────────────────────────────────
    frames = sorted(f.stem for f in INTENSITY_DIR.glob("frame_*.png"))
    if not frames:
        print("ERROR: 프레임 없음", flush=True)
        return 1

    print(f"\n[3] Detection + PCD 분리: {len(frames)} 프레임", flush=True)
    print("-" * 70, flush=True)

    all_det_summaries = []
    all_instance_plys = []

    for fname in frames:
        summary, inst_plys = process_frame_detection(fname, inferencer, OUTPUT_DIR)
        all_det_summaries.append(summary)
        all_instance_plys.extend(inst_plys)

        if "error" in summary:
            print(f"  {fname}: ERROR - {summary['error']}", flush=True)
            continue

        n_det   = summary["num_detected"]
        n_pcd   = summary["num_with_pcd"]
        vld_pct = summary["valid_mask_ratio"] * 100
        print(f"  {fname}: detected={n_det}, with_pcd={n_pcd}, "
              f"frame_valid={vld_pct:.1f}%", flush=True)

        for inst in summary["instances"]:
            if "skipped" in inst:
                print(f"    #{inst['instance_id']}: SKIPPED ({inst['skipped']}, "
                      f"{inst['num_points']} pts)", flush=True)
                continue
            c = inst["center_mm"]; s = inst["size_mm"]
            print(f"    #{inst['instance_id']}: score={inst['score']:.3f}, "
                  f"{inst['num_points_3d']} pts, "
                  f"center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm, "
                  f"size=({s[0]:.1f}, {s[1]:.1f}, {s[2]:.1f}) mm", flush=True)

    # Detection 중간 요약
    print("-" * 70, flush=True)
    total_det = sum(s.get("num_detected", 0) for s in all_det_summaries)
    total_pcd = sum(s.get("num_with_pcd", 0) for s in all_det_summaries)
    print(f"  Detection 완료: {len(frames)} 프레임, "
          f"총 검출 {total_det}개, PCD 추출 {total_pcd}개", flush=True)

    # ── [4] ICP 정합 ─────────────────────────────────────────────────────────
    if not all_instance_plys:
        print("\nERROR: ICP할 인스턴스 PLY 없음", flush=True)
        return 1

    print(f"\n[4] ICP 정합: {len(all_instance_plys)}개 인스턴스", flush=True)
    print("-" * 70, flush=True)

    all_icp_results = []
    n_success, n_fail = 0, 0

    for ply_path in all_instance_plys:
        result = process_instance_icp(ply_path, cad_pcd, cad_down, OUTPUT_DIR)
        all_icp_results.append(result)
        if "error" in result:
            n_fail += 1
        else:
            n_success += 1

    with (OUTPUT_DIR / "icp_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "input_data": input_data,
            "cad_path": str(CAD_PATH),
            "cad_axis_correction_deg": list(CAD_AXIS_CORRECTION_DEG),
            "voxel_size_cad_m": VOXEL_SIZE_CAD,
            "voxel_size_scene_m": VOXEL_SIZE_SCENE,
            "icp_stages": ICP_STAGES,
            "icp_fitness_threshold": ICP_FITNESS_THRESHOLD,
            "pick_offset_mm": [PICK_OFFSET_X_MM, PICK_OFFSET_Y_MM, PICK_OFFSET_Z_MM],
            "total": len(all_instance_plys),
            "success": n_success,
            "failed": n_fail,
            "results": all_icp_results,
        }, f, indent=2, ensure_ascii=False)

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print("-" * 70, flush=True)
    print(f"\n[5] 최종 요약", flush=True)
    print(f"  처리 프레임:       {len(frames)}", flush=True)
    print(f"  총 검출:           {total_det}", flush=True)
    print(f"  PCD 추출 성공:     {total_pcd}", flush=True)
    print(f"  ICP 정합 성공:     {n_success}", flush=True)
    print(f"  ICP 정합 실패:     {n_fail}", flush=True)
    print(f"\n  결과 위치: {OUTPUT_DIR}", flush=True)
    print(f"  - *_overlay.png:       2D detection 시각화", flush=True)
    print(f"  - *_colored.ply:       전체 PCD (배경 회색 + 인스턴스 컬러)", flush=True)
    print(f"  - *_obj{{i}}.ply:        인스턴스 단독 PCD", flush=True)
    print(f"  - *_summary.json:      detection 통계", flush=True)
    print(f"  - *_obj{{i}}_icp_vis.ply: ICP 정합 시각화 (scene+CAD+픽포인트)", flush=True)
    print(f"  - *_obj{{i}}_pose.json:   6DoF 자세 + 픽포인트", flush=True)
    print(f"  - icp_summary.json:    ICP 전체 요약", flush=True)
    print(f"\n  ✓ 파이프라인 완료", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())