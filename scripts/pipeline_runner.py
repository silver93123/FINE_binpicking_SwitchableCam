"""
pipeline_runner.py — BENIROBO UI ↔ FINE_RTMDet_v2 백엔드 파이프라인 실행기.

Run 탭의 "파이프라인 시작"을 누르면 이 QThread가:
  1) RTMDet-Ins 모델 로드
  2) CAD 모델 로드 (ICP 정합 기준)
  3) 카메라 연결 + 워밍업
  4) TCP 서버 오픈 (로봇이 클라이언트로 접속하는 구조 — FAIRINO 쪽에서 접속)
  5) 로봇이 'C' 명령을 보낼 때마다:
       캡처 → RTMDet 검출 → ICP 정합 → 픽포인트 계산 → 로봇에 TCP 응답 전송
       + UI에는 overlay 이미지 / 포인트클라우드 / 통계를 시그널로 전달
  6) "중지"를 누르면(stop()) 서버·카메라를 안전하게 정리하고 스레드 종료

핵심 수식(ICP 다단계 정합, 뒤집힘 보정, 포즈/픽포인트 계산)은
FINE_RTMDet_v2/scripts/5_Run_binpicking_TCP_v2.py 그대로 이식했습니다.
(다른 점: 인스턴스별 임시 PLY를 디스크에 쓰지 않고 메모리에서 바로 처리,
 그리고 각 프레임 처리 결과를 Qt 시그널로도 내보냄)

사용 예 (MainWindow 쪽):
    cfg = PipelineConfig(
        camera_cfg=map_ui_camera_config_to_backend(ui_cfg),
        rtmdet_config=Path(".../rtmdet-ins_bracket.py"),
        rtmdet_checkpoint=Path(".../best_xxx.pth"),
        cad_path=Path(".../bracket_v2.stl"),
    )
    runner = PipelineRunner(cfg)
    runner.log_message.connect(self.log.push)
    runner.image_ready.connect(self.center.run_page.show_image)
    runner.pointcloud_ready.connect(self.center.run_page.show_pointcloud)
    runner.result_ready.connect(self.center.run_page.show_result)
    runner.error.connect(lambda m: self.log.push(m, "ERR"))
    runner.finished_ok.connect(self._on_pipeline_finished)
    runner.start()
    ...
    runner.stop()   # "중지" 버튼
"""

from __future__ import annotations

import copy
import json
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    # 카메라 — src.camera.create_camera() 에 그대로 전달되는 dict
    # (vision_ui.map_ui_camera_config_to_backend() 결과를 그대로 넣으면 됨)
    camera_cfg: dict
    warmup_frames: int = 3

    # RTMDet
    rtmdet_config: Path = None
    rtmdet_checkpoint: Path = None
    device: str = "cuda:0"
    score_threshold: float = 0.3
    mask_iou_threshold: float = 0.6      # 중복 검출 제거 IoU 임계값
    min_points_per_instance: int = 100   # 이보다 점이 적으면 ICP 시도 안 함

    # CAD / ICP
    cad_path: Path = None
    cad_sample_points: int = 20000
    voxel_size_cad: float = 0.002
    voxel_size_scene: float = 0.003
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 1.5
    icp_stages: list = field(default_factory=lambda: [
        {"max_dist": 0.020, "max_iter": 100},
        {"max_dist": 0.010, "max_iter": 100},
        {"max_dist": 0.005, "max_iter": 100},
    ])
    icp_fitness_threshold: float = 0.5
    xyz_max_m: float = 2.0
    cad_axis_correction_deg: tuple = (-90.0, 90.0, 90.0)

    # 픽포인트 (CAD 로컬좌표계, m 단위 동차좌표 + mm 오프셋)
    cad_pick_local: tuple = (0.0, -0.100, 0.031, 1.0)
    pick_offset_mm: tuple = (-5.0, 0.0, 0.0)

    # TCP (로봇이 클라이언트로 접속)
    tcp_host: str = "0.0.0.0"
    tcp_port: int = 29999

    # 출력 경로
    out_dir: Path = field(default_factory=lambda: Path("data/captures/live"))


# ══════════════════════════════════════════════════════════════════
# 색상 팔레트 (인스턴스 구분용, BGR)
# ══════════════════════════════════════════════════════════════════
_PALETTE_BGR = np.array([
    [50, 50, 255], [50, 200, 50], [255, 100, 50],
    [30, 180, 255], [230, 50, 180], [200, 200, 30],
], dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════
# PipelineRunner
# ══════════════════════════════════════════════════════════════════
class PipelineRunner(QThread):
    log_message      = pyqtSignal(str, str)     # (msg, level)
    image_ready      = pyqtSignal(str)          # overlay PNG 경로
    pointcloud_ready = pyqtSignal(object)        # (N,3) float ndarray, mm
    frame_started    = pyqtSignal(int)           # frame_idx — 캡처 시작 시 emit ("Processing" 표시용)
    result_ready     = pyqtSignal(dict)          # 모니터 패널용 상세 info dict (아래 _process_one_frame 참고)
    client_status    = pyqtSignal(bool, str)     # (연결됨?, 주소)
    error            = pyqtSignal(str)
    finished_ok      = pyqtSignal()

    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg
        self._stop_requested = False
        self._server_sock: Optional[socket.socket] = None
        self._frame_idx = 0

    # ── 외부에서 호출 (MainWindow 쪽에서, "중지" 버튼) ─────────────
    def stop(self):
        self._stop_requested = True
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass

    def _log(self, msg: str, level: str = "INFO"):
        self.log_message.emit(msg, level)

    # ── 스레드 진입점 ────────────────────────────────────────────
    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished_ok.emit()

    def _run_impl(self):
        cfg = self.cfg
        if cfg.cad_path is None or not Path(cfg.cad_path).exists():
            raise FileNotFoundError(f"CAD 파일을 찾을 수 없습니다: {cfg.cad_path}")
        if cfg.rtmdet_config is None or cfg.rtmdet_checkpoint is None:
            raise FileNotFoundError("RTMDet config/checkpoint 경로가 설정되지 않았습니다.")

        import open3d as o3d
        import cv2
        from src.camera import create_camera
        from src.detection import RTMDetInferencer

        # [1] RTMDet 로드
        self._log(f"[1] RTMDet 모델 로드 중... ({Path(cfg.rtmdet_checkpoint).name})")
        inferencer = RTMDetInferencer(
            config=cfg.rtmdet_config,
            checkpoint=cfg.rtmdet_checkpoint,
            device=cfg.device,
            score_threshold=cfg.score_threshold,
        )
        self._log(f"    클래스: {inferencer.class_names}", "OK")

        # [2] CAD 로드
        self._log("[2] CAD 모델 로드 중...")
        cad_pcd = self._load_cad_as_pcd(o3d, cfg.cad_path, cfg.cad_sample_points,
                                         cfg.cad_axis_correction_deg)
        cad_down = cad_pcd.voxel_down_sample(cfg.voxel_size_cad)
        self._log(f"    {len(cad_pcd.points)}pts → 다운샘플 {len(cad_down.points)}pts", "OK")

        # [3] 카메라 연결 + 워밍업
        self._log(f"[3] 카메라 연결 + 워밍업 ({cfg.warmup_frames} frames)...")
        with create_camera(cfg.camera_cfg) as cam:
            for i in range(cfg.warmup_frames):
                cam.capture()
                self._log(f"    워밍업 {i + 1}/{cfg.warmup_frames}")
            self._log("카메라 준비 완료", "OK")

            # [4] TCP 서버 오픈
            self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_sock.bind((cfg.tcp_host, cfg.tcp_port))
            self._server_sock.listen(1)
            self._server_sock.settimeout(0.5)   # stop() 폴링 주기
            self._log(f"[4] TCP 대기 중: {cfg.tcp_host}:{cfg.tcp_port}", "OK")

            self._accept_loop(cam, inferencer, cad_pcd, cad_down, o3d, cv2)

        self._log("파이프라인 종료됨.")

    # ── TCP accept 루프 (stop_requested 폴링) ──────────────────────
    def _accept_loop(self, cam, inferencer, cad_pcd, cad_down, o3d, cv2):
        cfg = self.cfg
        while not self._stop_requested:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break   # stop()에서 소켓을 닫은 경우

            self.client_status.emit(True, str(addr))
            self._log(f"클라이언트 연결됨: {addr}", "OK")
            conn.settimeout(0.5)

            try:
                self._command_loop(conn, cam, inferencer, cad_pcd, cad_down, o3d, cv2)
            finally:
                conn.close()
                self.client_status.emit(False, "")

    def _command_loop(self, conn, cam, inferencer, cad_pcd, cad_down, o3d, cv2):
        while not self._stop_requested:
            try:
                cmd = self._recv_command(conn)
            except socket.timeout:
                continue
            if not cmd:
                self._log("클라이언트 연결 끊김")
                return
            self._log(f"수신: '{cmd}'")

            if cmd == "QUIT":
                conn.sendall(b"{'ok', 'bye'}\n")
                return

            if cmd == "C":
                self._frame_idx += 1
                try:
                    payload, extras = self._process_one_frame(
                        cam, inferencer, cad_pcd, cad_down, o3d, cv2)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    payload, extras = {"status": "error", "message": str(e)}, None

                msg = self._format_response(payload)
                conn.sendall((msg + "\n").encode("utf-8"))
                self._log(f"응답 전송: {msg}", "OK")

                if extras is not None:
                    self.image_ready.emit(extras["overlay_path"])
                    if extras.get("points_mm") is not None:
                        self.pointcloud_ready.emit(extras["points_mm"])
                    self.result_ready.emit(extras["result_summary"])
            else:
                conn.sendall(b"{'error', 'unknown command'}\n")

    @staticmethod
    def _recv_command(conn: socket.socket) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(1024)   # conn.settimeout(0.5) → 타임아웃 시 socket.timeout
            if not chunk:
                return ""
            buf += chunk
        return buf.decode("utf-8").strip()

    # ══════════════════════════════════════════════════════════
    # 한 프레임 처리: 캡처 → Detection → ICP → 픽포인트 → 응답 조립
    # ══════════════════════════════════════════════════════════
    def _process_one_frame(self, cam, inferencer, cad_pcd, cad_down, o3d, cv2):
        import time as _time
        cfg = self.cfg
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log(f"[frame_{self._frame_idx:04d}] 캡처 중... {now}")
        self.frame_started.emit(self._frame_idx)

        t0 = _time.perf_counter()
        frame = cam.capture()
        capture_ms = (_time.perf_counter() - t0) * 1000.0

        gray = frame.intensity
        pcd_organized = frame.points_organized.astype(np.float32)
        valid_mask = frame.valid_mask.astype(bool)
        bgr = np.stack([gray, gray, gray], axis=-1)

        valid_cnt = int(valid_mask.sum()); total = valid_mask.size
        valid_ratio = 100.0 * valid_cnt / total if total else 0.0
        pts_all = frame.points
        if pts_all.size:
            z_min, z_max = float(pts_all[:, 2].min()), float(pts_all[:, 2].max())
        else:
            z_min = z_max = float("nan")

        info_base = {
            "frame_idx": self._frame_idx,
            "timestamp": now,
            "capture_ms": capture_ms,
            "valid_ratio": valid_ratio,
            "z_min": z_min, "z_max": z_max,
        }

        # ── Detection ────────────────────────────────────────────
        t0 = _time.perf_counter()
        results = inferencer.infer(bgr)
        results, nms_removed = self._mask_nms(results, cfg.mask_iou_threshold)
        det_ms = (_time.perf_counter() - t0) * 1000.0
        for rem, winner, iou in nms_removed:
            self._log(f"  [NMS] score={rem.score:.2f} 제거 (IoU={iou:.2f})")
        self._log(f"검출: {len(results)}개")

        overlay = self._overlay_results(bgr, results, valid_mask)

        out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        dt_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_name = f"result_{dt_name}"

        if not results:
            overlay_path = out_dir / f"{frame_name}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            info = {**info_base, "status": "No detection", "det_ms": det_ms, "icp_ms": 0.0,
                    "num_detected": 0, "num_icp_ok": 0, "picks": []}
            return {"status": "No"}, {
                "overlay_path": str(overlay_path),
                "points_mm": frame.points if frame.points.size else None,
                "result_summary": info,
            }

        # ── 인스턴스별 ICP 정합 ───────────────────────────────────
        t0 = _time.perf_counter()
        icp_results = []   # [(DetectionResult, icp_result_dict)]
        picks_2d = []      # overlay 텍스트용

        for r in results:
            combined = r.mask & valid_mask
            obj_pts = pcd_organized[combined]
            if len(obj_pts) < cfg.min_points_per_instance:
                continue

            res = self._icp_one_instance(obj_pts, cad_down, o3d, cfg)
            if res is None:
                continue

            T, fit, rmse, flipped = res
            pose = self._transform_to_pose(T)
            pick = self._compute_pick_point(T, cfg)
            icp_results.append((r, {
                "icp_fitness": fit, "icp_rmse_m": rmse, "was_flipped": flipped,
                "pose": pose, "pick_point": pick,
            }))
            cx = float((r.bbox[0] + r.bbox[2]) / 2)
            cy = float((r.bbox[1] + r.bbox[3]) / 2)
            picks_2d.append((cx, cy, pick, fit, r.bbox))

        icp_ms = (_time.perf_counter() - t0) * 1000.0

        overlay_final = self._draw_picks_on_overlay(overlay, picks_2d, cv2) if picks_2d else overlay
        overlay_path = out_dir / f"{frame_name}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay_final)

        result_json_path = out_dir / f"{frame_name}_result.json"
        with result_json_path.open("w", encoding="utf-8") as f:
            json.dump({
                "frame": frame_name,
                "num_detected": len(results),
                "num_success": len(icp_results),
                "instances": [
                    {"class": r.class_name, "score": r.score, **info}
                    for r, info in icp_results
                ],
            }, f, indent=2, ensure_ascii=False)

        if not icp_results:
            info = {**info_base, "status": "No detection", "det_ms": det_ms, "icp_ms": icp_ms,
                    "num_detected": len(results), "num_icp_ok": 0, "picks": []}
            return {"status": "No"}, {
                "overlay_path": str(overlay_path),
                "points_mm": frame.points if frame.points.size else None,
                "result_summary": info,
            }

        # ── 응답 조립 (fitness 높은 순 정렬 → 로봇에는 전부 전달) ──
        icp_results.sort(key=lambda t: t[1]["icp_fitness"], reverse=True)
        picks = [{
            "position_mm": info["pick_point"]["position_mm"],
            "approach_deg": info["pick_point"]["approach_deg"],
            "icp_fitness": info["icp_fitness"],
            "class_name": r.class_name,
            "score": r.score,
        } for r, info in icp_results]

        best_pos = picks[0]["position_mm"]
        self._log(
            f"픽포인트: ({best_pos[0]:.1f}, {best_pos[1]:.1f}, {best_pos[2]:.1f}) mm  "
            f"fit={picks[0]['icp_fitness']:.3f}  (총 {len(picks)}개 중 최상)", "OK"
        )

        info = {**info_base, "status": "Done", "det_ms": det_ms, "icp_ms": icp_ms,
                "num_detected": len(results), "num_icp_ok": len(icp_results), "picks": picks}

        return {"status": "ok", "picks": picks}, {
            "overlay_path": str(overlay_path),
            "points_mm": frame.points if frame.points.size else None,
            "result_summary": info,
        }

    # ══════════════════════════════════════════════════════════
    # ICP (수식은 5_Run_binpicking_TCP_v2.py와 동일)
    # ══════════════════════════════════════════════════════════
    def _icp_one_instance(self, obj_pts, cad_down, o3d, cfg):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pts / 1000.0)
        sc, _ = pcd.remove_statistical_outlier(cfg.outlier_nb_neighbors, cfg.outlier_std_ratio)
        sd = sc.voxel_down_sample(cfg.voxel_size_scene)
        if len(sd.points) < 20:
            return None

        T_init = np.eye(4)
        T_init[:3, 3] = np.asarray(sd.get_center()) - np.asarray(cad_down.get_center())

        T, fit, rmse = self._run_icp_multistage(cad_down, sd, T_init, o3d, cfg.icp_stages)
        T, fit, rmse, flipped = self._correct_flipped_pose(T, cad_down, sd, o3d, cfg.icp_stages)

        if fit < cfg.icp_fitness_threshold:
            self._log(f"  ICP 실패 (fitness={fit:.4f})")
            return None
        if max(abs(v) for v in T[:3, 3]) > cfg.xyz_max_m:
            self._log("  xyz 범위 이상")
            return None
        return T, fit, rmse, flipped

    @staticmethod
    def _run_icp_multistage(src, tgt, T_init, o3d, stages):
        T = T_init.copy()
        for stage in stages:
            res = o3d.pipelines.registration.registration_icp(
                src, tgt, stage["max_dist"], T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage["max_iter"]),
            )
            T = np.asarray(res.transformation)
        final = o3d.pipelines.registration.evaluate_registration(src, tgt, stages[-1]["max_dist"], T)
        return T, float(final.fitness), float(final.inlier_rmse)

    @classmethod
    def _correct_flipped_pose(cls, T, src, tgt, o3d, stages):
        if T[:3, :3][2, 2] >= 0:
            final = o3d.pipelines.registration.evaluate_registration(src, tgt, stages[-1]["max_dist"], T)
            return T, float(final.fitness), float(final.inlier_rmse), False
        R_flip = np.diag([-1.0, -1.0, 1.0])
        T_flip = np.eye(4); T_flip[:3, :3] = R_flip
        c = T[:3, 3]; T_flip[:3, 3] = c - R_flip @ c
        T_f, fit, rmse = cls._run_icp_multistage(src, tgt, T_flip @ T, o3d, stages)
        return T_f, fit, rmse, True

    @staticmethod
    def _transform_to_pose(T):
        xyz_mm = (T[:3, 3] * 1000.0).tolist()
        R = T[:3, :3]
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        cp = np.cos(pitch)
        if abs(cp) > 1e-6:
            roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
            yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
        else:
            roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
        e = np.degrees([roll, pitch, yaw]).tolist()
        return {
            "xyz_mm": [round(v, 3) for v in xyz_mm],
            "euler_deg": {"roll_deg": round(e[0], 4), "pitch_deg": round(e[1], 4), "yaw_deg": round(e[2], 4)},
        }

    @staticmethod
    def _compute_pick_point(T, cfg: PipelineConfig):
        pl = np.array(cfg.cad_pick_local, dtype=float).copy()
        off = cfg.pick_offset_mm
        pl[0] += off[0] / 1000.0
        pl[1] += off[1] / 1000.0
        pl[2] += off[2] / 1000.0
        wt = T @ pl
        pos = (wt[:3] * 1000.0).tolist()
        R = T[:3, :3]
        pitch = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))))
        cp = np.cos(np.radians(pitch))
        if abs(cp) > 1e-6:
            roll = float(np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp)))
            yaw = float(np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp)))
        else:
            roll, yaw = 0.0, float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))
        return {
            "position_mm": [round(v, 3) for v in pos],
            "approach_deg": {"roll_deg": round(roll, 4), "pitch_deg": round(pitch, 4), "yaw_deg": round(yaw, 4)},
        }

    @staticmethod
    def _load_cad_as_pcd(o3d, cad_path, sample_points, axis_correction_deg):
        def Rx(d):
            c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
            R = np.eye(3); R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
            return R

        def Ry(d):
            c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
            R = np.eye(3); R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c
            return R

        def Rz(d):
            c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
            R = np.eye(3); R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
            return R

        mesh = o3d.io.read_triangle_mesh(str(cad_path))
        ext = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
        if ext.max() > 10.0:   # mm 단위로 저장된 CAD → m로 변환
            mesh.scale(1.0 / 1000.0, center=np.zeros(3))
        rx, ry, rz = axis_correction_deg
        R = Rz(rz) @ Ry(ry) @ Rx(rx)
        center = np.asarray(mesh.get_center())
        T_fix = np.eye(4); T_fix[:3, :3] = R; T_fix[:3, 3] = center - R @ center
        mesh.transform(T_fix)
        return mesh.sample_points_poisson_disk(sample_points)

    # ══════════════════════════════════════════════════════════
    # Detection 후처리 (NMS / overlay)
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _mask_nms(results, iou_threshold):
        keep, removed = [], []
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
                rj = results[j]
                inter = (ri.mask & rj.mask).sum()
                if inter == 0:
                    continue
                area_j = rj.mask.sum()
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    suppressed[j] = True
                    removed.append((rj, ri, float(iou)))
        return keep, removed

    @staticmethod
    def _overlay_results(image_bgr, results, valid_mask=None):
        import cv2
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

    @staticmethod
    def _draw_picks_on_overlay(image_bgr, picks_2d, cv2):
        out = image_bgr.copy()
        H, W = out.shape[:2]
        for i, (px, py, pick, fit, bbox) in enumerate(picks_2d):
            color = tuple(int(c) for c in _PALETTE_BGR[i % len(_PALETTE_BGR)])
            pp = pick["position_mm"]
            cv2.drawMarker(out, (int(px), int(py)), color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            line1 = f"#{i}  ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm"
            line2 = f"ICP fit: {fit:.3f}"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, thick)
            box_w, box_h = max(w1, w2) + 8, h1 + h2 + 12
            x1, y1, x2, y2 = [int(v) for v in bbox]
            tx, ty = max(x1, 0), y1 - box_h - 4
            if ty < 0:
                ty = y2 + 4
            ty = min(ty, H - box_h - 2); tx = min(tx, W - box_w - 2)
            cv2.rectangle(out, (tx - 2, ty), (tx + box_w, ty + box_h), (0, 0, 0), -1)
            cv2.putText(out, line1, (tx + 2, ty + h1 + 2), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(out, line2, (tx + 2, ty + h1 + h2 + 8), font, scale, (200, 200, 200), thick, cv2.LINE_AA)
        return out

    # ══════════════════════════════════════════════════════════
    # 로봇 TCP 응답 포맷
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _format_response(payload: dict) -> str:
        status = payload.get("status")
        if status == "ok":
            parts = ["'ok'", str(len(payload["picks"]))]
            for pk in payload["picks"]:
                pp, deg = pk["position_mm"], pk["approach_deg"]
                fit = round(pk["icp_fitness"], 2)
                tup = (round(pp[0], 3), round(pp[1], 3), round(pp[2], 3),
                       round(deg["roll_deg"], 3), round(deg["pitch_deg"], 3),
                       round(deg["yaw_deg"], 3), fit)
                parts.append(str(tup))
            return "{" + ", ".join(parts) + "}"
        if status in ("no_object", "No"):
            return "{'No'}"
        msg = payload.get("message", "unknown error")
        return "{" + f"'error', '{msg}'" + "}"