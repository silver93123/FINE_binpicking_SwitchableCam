"""Stage 5-B: 인스턴스 PCD ↔ CAD 모델 ICP 정합 → 6DoF 자세 추정.

파이프라인:
    1. STL CAD 모델 로드 → mm→m 변환 → 축 보정 회전 (Rx=-90, Ry=90, Rz=90)
    2. 인스턴스 PCD 전처리 (노이즈 제거 → 다운샘플 → 노말)
    3. CAD를 scene 중심으로 초기 정렬 (RANSAC 탐색 범위 최소화)
    4. FPFH + RANSAC 글로벌 정합
    5. ICP 정밀 정합
    6. 6DoF 자세 추출 (xyz mm + 오일러각 deg)
    7. 정합 결과 PLY 시각화 저장

축 보정 배경:
    STL 좌표계와 센서 좌표계가 달라 CAD가 세워진 채로 로드됨.
    Rx=-90 → Ry=90 → Rz=90 적용 시 scene과 방향 일치 확인됨.

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/7_stage5b_icp.py

입력:
    data/inference_results/{input_data}/frame_NNNN_obj{i}.ply
    data/cad/bracket_v2.stl

출력:
    data/inference_results/{input_data}/
    ├── frame_NNNN_obj{i}_pose.json
    └── frame_NNNN_obj{i}_icp_vis.ply
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

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

input_data = "20260519_0000"

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
CAD_PATH = ROOT / "data" / "cad" / "bracket_v3.stl"
INFERENCE_DIR = ROOT / "data" / "inference_results" / input_data
OUTPUT_DIR = INFERENCE_DIR

CAD_SAMPLE_POINTS = 30000

VOXEL_SIZE_CAD   = 0.0015       # 1.5mm
VOXEL_SIZE_SCENE = 0.003        # 3.0mm

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

CAMERA_LOCATION = np.array([0.0, 0.0, 2.0])

FPFH_RADIUS_NORMAL  = VOXEL_SIZE_SCENE * 3     # 9mm
FPFH_RADIUS_FEATURE = VOXEL_SIZE_SCENE * 8     # 24mm

RANSAC_DISTANCE   = VOXEL_SIZE_SCENE * 3.0     # 9mm
RANSAC_ITER       = 100000  # 단일 시도당 반복 수 (다중 시도로 보완)
RANSAC_ATTEMPTS   = 5       # 대칭 형상 대응: 여러 번 시도해서 best 선택
RANSAC_CONFIDENCE = 0.999

ICP_MAX_ITER          = 300
ICP_DISTANCE          = VOXEL_SIZE_SCENE * 2.0  # 6mm
ICP_FITNESS_THRESHOLD = 0.3

XYZ_MAX_M = 2.0  # 센서 범위 밖이면 로컬 미니멈으로 판정

# CAD 축 보정 — STL 좌표계 → 센서 좌표계
# 시각적으로 확인된 값: Rx=-90, Ry=90, Rz=90
CAD_AXIS_CORRECTION_DEG = (-90, 90, 90)   # (rx, ry, rz) 단위: deg


# -----------------------------------------------------------------------------
# 유틸: 회전행렬
# -----------------------------------------------------------------------------

def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)]])

def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [0,          1, 0         ],
                     [-np.sin(a), 0, np.cos(a)]])

def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,          1]])


def axis_correction_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """CAD 자체 중심 기준 Rz @ Ry @ Rx 순서의 4x4 회전행렬 반환.

    중심 고정 회전: T[:3,3] = center - R @ center
    """
    R = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    T = np.eye(4)
    T[:3, :3] = R
    return T   # 중심 고정은 load_cad_as_pcd에서 처리


# -----------------------------------------------------------------------------
# CAD 모델 준비
# -----------------------------------------------------------------------------

def load_cad_as_pcd(stl_path: Path, n_points: int = CAD_SAMPLE_POINTS) -> o3d.geometry.PointCloud:
    """STL 로드 → mm→m 변환 → 축 보정 회전 → 포인트 샘플링.

    축 보정은 CAD 자체 중심 기준으로 적용하여 위치가 흩어지지 않게 함.
    """
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if not mesh.has_triangles():
        raise ValueError(f"STL 로드 실패: {stl_path}")

    # mm → m
    extent_before = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    STL 원본 extent: {np.round(extent_before, 2)} mm", flush=True)
    mesh.scale(1.0 / 1000.0, center=np.zeros(3))
    extent_after = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    STL 변환 후 extent: {np.round(extent_after, 4)} m", flush=True)

    # 축 보정 회전 — CAD 자체 중심 기준
    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    center = np.asarray(mesh.get_center())

    T_fix = np.eye(4)
    T_fix[:3, :3] = R
    T_fix[:3, 3] = center - R @ center   # 중심 고정

    mesh.transform(T_fix)
    print(f"    축 보정 적용: Rx={rx}° Ry={ry}° Rz={rz}°", flush=True)
    extent_fix = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    보정 후 extent: {np.round(extent_fix, 4)} m", flush=True)

    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(n_points)

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=FPFH_RADIUS_NORMAL, max_nn=30
            )
        )
    return pcd


# -----------------------------------------------------------------------------
# 전처리
# -----------------------------------------------------------------------------

def preprocess_cad(
    pcd: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd_down = pcd.voxel_down_sample(VOXEL_SIZE_CAD)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=FPFH_RADIUS_NORMAL, max_nn=30
        )
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=10)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS_FEATURE, max_nn=100),
    )
    return pcd_down, fpfh


def preprocess_scene(
    pcd: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature, int, int]:
    n_before = len(np.asarray(pcd.points))
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO,
    )
    n_after = len(np.asarray(pcd_clean.points))

    pcd_down = pcd_clean.voxel_down_sample(VOXEL_SIZE_SCENE)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=FPFH_RADIUS_NORMAL, max_nn=30
        )
    )
    pcd_down.orient_normals_towards_camera_location(camera_location=CAMERA_LOCATION)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS_FEATURE, max_nn=100),
    )
    return pcd_down, fpfh, n_before, n_after


# -----------------------------------------------------------------------------
# 중심 정렬
# -----------------------------------------------------------------------------

def center_align_transform(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> np.ndarray:
    """source 중심 → target 중심으로 이동하는 4x4 평행이동 행렬."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(target.get_center()) - np.asarray(source.get_center())
    return T


# -----------------------------------------------------------------------------
# 글로벌 정합 (RANSAC)
# -----------------------------------------------------------------------------

def global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=RANSAC_DISTANCE,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DISTANCE),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            RANSAC_ITER, RANSAC_CONFIDENCE
        ),
    )


# -----------------------------------------------------------------------------
# ICP
# -----------------------------------------------------------------------------

def refine_with_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_icp(
        source, target,
        ICP_DISTANCE, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
    )



# -----------------------------------------------------------------------------
# 뒤집힘 보정
# -----------------------------------------------------------------------------

def correct_flipped_pose(
    T: np.ndarray,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, bool]:
    """ICP 결과가 뒤집힌 자세인 경우 Z축 180도 회전 후 ICP 재수렴.

    판정: R[2,2] < 0 이면 CAD Z축이 아래를 향함 = 뒤집힌 자세.
    센서가 위에서 찍으므로 브라켓 상단면이 +Z를 향해야 정상.

    Returns:
        (보정된 T, 보정 여부)
    """
    R = T[:3, :3]
    if R[2, 2] >= 0:
        return T, False  # 정상 자세

    # Z축 180도 회전 (translation 위치 중심 고정)
    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4)
    T_flip[:3, :3] = R_flip
    c = T[:3, 3]
    T_flip[:3, 3] = c - R_flip @ c
    T_flipped = T_flip @ T

    # 뒤집은 초기값으로 ICP 재수렴
    icp = refine_with_icp(source, target, T_flipped)
    return np.asarray(icp.transformation), True

# -----------------------------------------------------------------------------
# 변환행렬 → 6DoF
# -----------------------------------------------------------------------------

def transform_to_pose(T: np.ndarray) -> dict:
    """4x4 변환행렬 (m 단위) → xyz mm + ZYX 오일러각 deg."""
    xyz_mm = (T[:3, 3] * 1000.0).tolist()

    R = T[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])

    euler_deg = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm": [round(v, 3) for v in xyz_mm],
        "euler_deg": {
            "roll_deg":  round(euler_deg[0], 4),
            "pitch_deg": round(euler_deg[1], 4),
            "yaw_deg":   round(euler_deg[2], 4),
        },
        "transform_matrix": T.tolist(),
    }


# -----------------------------------------------------------------------------
# 시각화
# -----------------------------------------------------------------------------

def save_icp_visualization(
    scene_pcd: o3d.geometry.PointCloud,
    cad_pcd: o3d.geometry.PointCloud,
    T: np.ndarray,
    out_path: Path,
) -> None:
    scene_vis = copy.deepcopy(scene_pcd)
    n = len(np.asarray(scene_vis.points))
    scene_vis.colors = o3d.utility.Vector3dVector(np.tile([0.6, 0.6, 0.6], (n, 1)))

    cad_vis = copy.deepcopy(cad_pcd)
    cad_vis.transform(T)
    n = len(np.asarray(cad_vis.points))
    cad_vis.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.9, 0.3], (n, 1)))

    o3d.io.write_point_cloud(str(out_path), scene_vis + cad_vis, write_ascii=False)


# -----------------------------------------------------------------------------
# 단일 인스턴스 처리
# -----------------------------------------------------------------------------

def process_instance(
    instance_ply: Path,
    cad_pcd: o3d.geometry.PointCloud,
    cad_down: o3d.geometry.PointCloud,
    cad_fpfh: o3d.pipelines.registration.Feature,
    output_dir: Path,
) -> dict:
    stem = instance_ply.stem

    scene_pcd = o3d.io.read_point_cloud(str(instance_ply))
    n_pts = len(np.asarray(scene_pcd.points))
    if n_pts < 50:
        return {"file": stem, "error": f"포인트 부족: {n_pts}개"}

    print(f"  {stem}: {n_pts} pts", flush=True)

    # scene 전처리
    scene_down, scene_fpfh, n_before, n_after = preprocess_scene(scene_pcd)
    removal_pct = (1 - n_after / max(n_before, 1)) * 100
    n_scene_down = len(np.asarray(scene_down.points))
    n_cad_down   = len(np.asarray(cad_down.points))
    print(f"    노이즈 제거: {n_before} → {n_after} pts ({removal_pct:.1f}% 제거)", flush=True)
    print(f"    다운샘플:    scene={n_scene_down}  cad={n_cad_down}", flush=True)

    normals = np.asarray(scene_down.normals)
    nz_mean = normals[:, 2].mean() if len(normals) > 0 else 0.0
    print(f"    노말 Z 평균: {nz_mean:.3f}  ({'✓' if nz_mean > 0.3 else '△ 불안정'})", flush=True)

    # 중심 정렬 → RANSAC
    T_center = center_align_transform(cad_down, scene_down)
    cad_down_shifted = copy.deepcopy(cad_down)
    cad_down_shifted.transform(T_center)

    src_c = np.round(np.asarray(cad_down.get_center()), 3)
    tgt_c = np.round(np.asarray(scene_down.get_center()), 3)
    print(f"    중심 정렬:   CAD {src_c} → scene {tgt_c}", flush=True)

    # RANSAC 다중 시도 → ICP → best 선택
    # 브라켓처럼 대칭에 가까운 형상은 RANSAC이 180° 뒤집힌 자세도 비슷한 fitness로 찾음.
    # 여러 번 시도해서 ICP fitness가 가장 높고 pitch가 정상 범위인 결과를 선택.
    print(f"    글로벌 정합(RANSAC) × {RANSAC_ATTEMPTS}회 시도 중...", flush=True)

    best_icp_result  = None
    best_T_final     = None
    best_icp_fitness = -1.0
    best_global_fitness = 0.0

    for attempt in range(RANSAC_ATTEMPTS):
        g = global_registration(cad_down_shifted, scene_down, cad_fpfh, scene_fpfh)
        if g.fitness < 0.05:
            print(f"      [{attempt+1}/{RANSAC_ATTEMPTS}] 글로벌 fitness={g.fitness:.4f} → 스킵", flush=True)
            continue

        T_ransac   = np.asarray(g.transformation)
        T_combined = T_ransac @ T_center
        icp = refine_with_icp(cad_down, scene_down, T_combined)
        T_icp = np.asarray(icp.transformation)

        # pitch 추출 (|pitch| > 45° 이면 뒤집힌 자세)
        R = T_icp[:3, :3]
        pitch_rad = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        pitch_deg = float(np.degrees(pitch_rad))
        xyz_ok = max(abs(v) for v in T_icp[:3, 3]) < XYZ_MAX_M
        pitch_ok = abs(pitch_deg) < 45.0

        flag = "✓" if (xyz_ok and pitch_ok) else ("△ pitch 이상" if not pitch_ok else "△ xyz 이상")
        print(f"      [{attempt+1}/{RANSAC_ATTEMPTS}] 글로벌={g.fitness:.3f} "
              f"ICP={icp.fitness:.3f} pitch={pitch_deg:.1f}° {flag}", flush=True)

        # pitch 정상 범위 우선, 그 안에서 ICP fitness 최대
        is_better = (
            icp.fitness > best_icp_fitness
            and xyz_ok
            and pitch_ok
        )
        if is_better:
            best_icp_result     = icp
            best_T_final        = T_icp
            best_icp_fitness    = icp.fitness
            best_global_fitness = g.fitness

    # pitch 정상인 결과가 없으면 xyz만 정상인 best로 폴백
    if best_T_final is None:
        print(f"    △ pitch 정상 결과 없음 → xyz 정상 결과로 폴백", flush=True)
        for attempt in range(RANSAC_ATTEMPTS):
            g = global_registration(cad_down_shifted, scene_down, cad_fpfh, scene_fpfh)
            if g.fitness < 0.05:
                continue
            T_combined = np.asarray(g.transformation) @ T_center
            icp = refine_with_icp(cad_down, scene_down, T_combined)
            T_icp = np.asarray(icp.transformation)
            if icp.fitness > best_icp_fitness and max(abs(v) for v in T_icp[:3, 3]) < XYZ_MAX_M:
                best_icp_result     = icp
                best_T_final        = T_icp
                best_icp_fitness    = icp.fitness
                best_global_fitness = g.fitness

    if best_T_final is None:
        print(f"    ✗ 글로벌 정합 실패 (전 시도)", flush=True)
        return {"file": stem, "error": "글로벌 정합 실패 (전 시도)"}

    fitness = best_icp_result.fitness
    rmse    = best_icp_result.inlier_rmse
    T_final = best_T_final

    # 뒤집힘 보정 — R[2,2] < 0 이면 Z축 180도 회전 후 ICP 재수렴
    T_final, was_flipped = correct_flipped_pose(T_final, cad_down, scene_down)
    if was_flipped:
        fitness = refine_with_icp(cad_down, scene_down, T_final).fitness
        print(f"    △ 뒤집힘 감지 → Z축 180도 보정 적용", flush=True)

    t_m  = np.round(T_final[:3, 3], 4)
    t_mm = np.round(T_final[:3, 3] * 1000, 1)
    print(f"    Best ICP fitness={fitness:.4f}, rmse={rmse:.6f}", flush=True)
    print(f"    T translation: {t_m} m = {t_mm} mm", flush=True)

    if fitness < ICP_FITNESS_THRESHOLD:
        print(f"    ✗ ICP 정합 실패 (fitness={fitness:.4f} < {ICP_FITNESS_THRESHOLD})", flush=True)
        return {"file": stem, "error": "ICP 정합 실패",
                "global_fitness": float(best_global_fitness),
                "icp_fitness": float(fitness), "icp_rmse": float(rmse)}

    if max(abs(v) for v in T_final[:3, 3]) > XYZ_MAX_M:
        print(f"    ✗ xyz 비정상: {t_m} m → 로컬 미니멈", flush=True)
        pose_tmp = transform_to_pose(T_final)
        return {"file": stem, "error": "xyz 범위 이상 (로컬 미니멈)",
                "icp_fitness": float(fitness), "xyz_mm": pose_tmp["xyz_mm"]}

    pose = transform_to_pose(T_final)
    xyz  = pose["xyz_mm"]

    vis_path = output_dir / f"{stem}_icp_vis.ply"
    save_icp_visualization(scene_pcd, cad_pcd, T_final, vis_path)

    result = {
        "file": stem,
        "input_ply": str(instance_ply.relative_to(ROOT)),
        "num_points_scene": n_pts,
        "num_points_after_outlier_removal": n_after,
        "global_fitness": float(best_global_fitness),
        "icp_fitness": float(fitness),
        "icp_rmse_m": float(rmse),
        "pose": pose,
        "vis_ply": str(vis_path.relative_to(ROOT)),
    }
    with (output_dir / f"{stem}_pose.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    eul = pose["euler_deg"]
    print(f"    ✓ 자세: xyz=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm  "
          f"roll={eul['roll_deg']:.1f}° pitch={eul['pitch_deg']:.1f}° "
          f"yaw={eul['yaw_deg']:.1f}°", flush=True)

    return result


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------

def main() -> int:
    for path, name in [(CAD_PATH, "CAD"), (INFERENCE_DIR, "추론 결과 폴더")]:
        if not path.exists():
            print(f"ERROR: {name} 없음: {path}", flush=True)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  Stage 5-B: ICP 정합 → 6DoF 자세 추정", flush=True)
    print("=" * 70, flush=True)
    print(f"  CAD:             {CAD_PATH}", flush=True)
    print(f"  CAD 축 보정:     Rx={CAD_AXIS_CORRECTION_DEG[0]}° "
          f"Ry={CAD_AXIS_CORRECTION_DEG[1]}° Rz={CAD_AXIS_CORRECTION_DEG[2]}°", flush=True)
    print(f"  Inference:       {INFERENCE_DIR}", flush=True)
    print(f"  Voxel CAD:       {VOXEL_SIZE_CAD * 1000:.1f} mm", flush=True)
    print(f"  Voxel scene:     {VOXEL_SIZE_SCENE * 1000:.1f} mm", flush=True)
    print(f"  Outlier (nn/std):{OUTLIER_NB_NEIGHBORS} / {OUTLIER_STD_RATIO}", flush=True)
    print(f"  FPFH normal r:   {FPFH_RADIUS_NORMAL * 1000:.1f} mm", flush=True)
    print(f"  FPFH feature r:  {FPFH_RADIUS_FEATURE * 1000:.1f} mm", flush=True)
    print(f"  RANSAC dist:     {RANSAC_DISTANCE * 1000:.1f} mm", flush=True)
    print(f"  ICP dist:        {ICP_DISTANCE * 1000:.1f} mm", flush=True)
    print(f"  Camera Z:        {CAMERA_LOCATION[2]:.1f} m", flush=True)
    print("=" * 70, flush=True)

    print("\n[1] CAD 모델 로드 및 전처리 중...", flush=True)
    try:
        cad_pcd = load_cad_as_pcd(CAD_PATH)
    except Exception as e:
        print(f"ERROR: CAD 로드 실패: {e}", flush=True)
        return 1
    print(f"    ✓ CAD 샘플 포인트: {len(np.asarray(cad_pcd.points))}개", flush=True)

    cad_down, cad_fpfh = preprocess_cad(cad_pcd)
    print(f"    ✓ CAD 다운샘플:    {len(np.asarray(cad_down.points))}개 "
          f"(voxel={VOXEL_SIZE_CAD * 1000:.1f}mm)", flush=True)
    print(f"    CAD center: {np.round(np.asarray(cad_down.get_center()), 4)} m", flush=True)

    instance_plys = sorted(INFERENCE_DIR.glob("frame_*_obj*.ply"))
    instance_plys = [
        p for p in instance_plys
        if "_colored" not in p.stem and "_icp_vis" not in p.stem
    ]
    if not instance_plys:
        print("\nERROR: 인스턴스 PLY 없음. Stage 5-A를 먼저 실행하세요.", flush=True)
        return 1

    print(f"\n[2] ICP 정합: {len(instance_plys)}개 인스턴스", flush=True)
    print("-" * 70, flush=True)

    all_results = []
    n_success, n_fail = 0, 0

    for ply_path in instance_plys:
        result = process_instance(ply_path, cad_pcd, cad_down, cad_fpfh, OUTPUT_DIR)
        all_results.append(result)
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
            "outlier_nb_neighbors": OUTLIER_NB_NEIGHBORS,
            "outlier_std_ratio": OUTLIER_STD_RATIO,
            "fpfh_radius_normal_m": FPFH_RADIUS_NORMAL,
            "fpfh_radius_feature_m": FPFH_RADIUS_FEATURE,
            "ransac_distance_m": RANSAC_DISTANCE,
            "icp_distance_m": ICP_DISTANCE,
            "icp_fitness_threshold": ICP_FITNESS_THRESHOLD,
            "camera_location": CAMERA_LOCATION.tolist(),
            "total": len(instance_plys),
            "success": n_success,
            "failed": n_fail,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print("-" * 70, flush=True)
    print(f"\n[3] 요약", flush=True)
    print(f"  전체 인스턴스:   {len(instance_plys)}", flush=True)
    print(f"  정합 성공:       {n_success}", flush=True)
    print(f"  정합 실패:       {n_fail}", flush=True)
    print(f"\n  결과 위치: {OUTPUT_DIR}", flush=True)
    print(f"  - *_pose.json:      6DoF 자세 (xyz mm + 오일러각 + 4x4 행렬)", flush=True)
    print(f"  - *_icp_vis.ply:    정합 시각화 (회색=scene, 초록=CAD)", flush=True)
    print(f"  - icp_summary.json: 전체 요약", flush=True)
    print(f"\n  Open3D에서 확인:", flush=True)
    print(f"    python -c \"import open3d as o3d; "
          f"o3d.visualization.draw_geometries("
          f"[o3d.io.read_point_cloud('{OUTPUT_DIR}/frame_0001_obj0_icp_vis.ply')])\"",
          flush=True)
    print(f"\n  ✓ Stage 5-B 완료", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())