"""Stage 4: 학습된 RTMDet-Ins로 브라켓 추론 + 결과 시각화.

용도:
    Stage 3에서 fine-tune한 모델을 로드해서 학습 데이터에 추론.
    학습이 제대로 됐는지 시각적으로 검증.

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/5_inference_bracket.py

출력:
    data/inference_results/bracket_v1/
    ├── frame_0001_overlay.png     ← 마스크 + bbox + score 합성
    ├── frame_0002_overlay.png
    └── ...
    + 콘솔에 검출 통계
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import RTMDetInferencer  # noqa: E402

input_data = "20260519_0000"
# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
CONFIG_PATH = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1" / "rtmdet-ins_bracket.py"
CHECKPOINT_PATH = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1" / "best_coco_bbox_mAP_epoch_50.pth"

INPUT_DIR = ROOT / "data" / "dataset" / input_data / "intensity"  ### 
OUTPUT_DIR = ROOT / "data" / "inference_results" / input_data

# 추론 임계값 (학습 데이터라 높게 설정 가능)
SCORE_THRESHOLD = 0.5


def overlay_results(image: np.ndarray, results, palette) -> np.ndarray:
    """이미지 위에 마스크 + bbox + 라벨 합성."""
    overlay = image.copy()

    # 1. 마스크 색칠 (alpha 0.4)
    for i, r in enumerate(results):
        color = palette[i % len(palette)]
        color_layer = np.zeros_like(overlay)
        color_layer[r.mask] = color
        overlay[r.mask] = (0.6 * overlay[r.mask] + 0.4 * color_layer[r.mask]).astype(np.uint8)

    # 2. BBox + 라벨
    for i, r in enumerate(results):
        color = tuple(int(c) for c in palette[i % len(palette)])
        x1, y1, x2, y2 = r.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{r.class_name} {r.score:.2f}"
        # 라벨 배경 박스
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def main() -> int:
    # 사전 검증
    if not CONFIG_PATH.exists():
        print(f"ERROR: config 없음: {CONFIG_PATH}", flush=True)
        return 1
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: checkpoint 없음: {CHECKPOINT_PATH}", flush=True)
        return 1
    if not INPUT_DIR.exists():
        print(f"ERROR: 입력 폴더 없음: {INPUT_DIR}", flush=True)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  Stage 4: 학습된 모델로 추론", flush=True)
    print("=" * 70, flush=True)
    print(f"  Config:     {CONFIG_PATH}", flush=True)
    print(f"  Checkpoint: {CHECKPOINT_PATH.name}", flush=True)
    print(f"  Input:      {INPUT_DIR}", flush=True)
    print(f"  Output:     {OUTPUT_DIR}", flush=True)
    print(f"  Threshold:  {SCORE_THRESHOLD}", flush=True)
    print("=" * 70, flush=True)

    # 모델 로드
    print("\n[1] 모델 로드 중...", flush=True)
    inferencer = RTMDetInferencer(
        config=CONFIG_PATH,
        checkpoint=CHECKPOINT_PATH,
        device="cuda:0",
        score_threshold=SCORE_THRESHOLD,
    )
    print(f"    ✓ 클래스: {inferencer.class_names}", flush=True)

    # 색상 팔레트 (인스턴스 구분용)
    palette = np.array([
        [50, 50, 255],     # 빨강
        [50, 200, 50],     # 초록
        [255, 100, 50],    # 청록
        [30, 180, 255],    # 주황
        [230, 50, 180],    # 자홍
        [200, 200, 30],    # 노랑
    ], dtype=np.uint8)

    # 이미지 목록
    images = sorted(INPUT_DIR.glob("frame_*.png"))
    if not images:
        print(f"ERROR: 입력 이미지 없음", flush=True)
        return 1

    print(f"\n[2] 추론 시작: {len(images)}장", flush=True)
    print("-" * 70, flush=True)

    all_counts = []
    all_scores = []

    for img_path in images:
        # mono → BGR 3채널 (RTMDet 입력 형식)
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  {img_path.name}: 읽기 실패 - 스킵", flush=True)
            continue
        bgr = np.stack([gray, gray, gray], axis=-1)

        # 추론
        results = inferencer.infer(bgr)

        # 시각화
        overlay = overlay_results(bgr, results, palette)
        out_path = OUTPUT_DIR / f"{img_path.stem}_overlay.png"
        cv2.imwrite(str(out_path), overlay)

        # 통계
        scores = [r.score for r in results]
        all_counts.append(len(results))
        all_scores.extend(scores)
        score_str = ", ".join(f"{s:.3f}" for s in scores) if scores else "(없음)"
        print(f"  {img_path.name}: {len(results)}개 검출  scores=[{score_str}]", flush=True)

    # 종합 요약
    print("-" * 70, flush=True)
    print(f"\n[3] 요약", flush=True)
    print(f"  처리 이미지:  {len(images)}장", flush=True)
    print(f"  총 검출:      {sum(all_counts)}개", flush=True)
    print(f"  이미지당 평균: {np.mean(all_counts):.1f}개", flush=True)
    if all_scores:
        print(f"  점수 통계:    mean={np.mean(all_scores):.3f}, "
              f"min={min(all_scores):.3f}, max={max(all_scores):.3f}", flush=True)

    # GT 비교 (라벨이 있으면)
    gt_path = ROOT / "data" / "dataset" / "brackets_v1" / "coco_labels" / "annotations" / "instances_Train.json"
    if gt_path.exists():
        import json
        with open(gt_path) as f:
            coco = json.load(f)
        gt_count = len(coco["annotations"])
        print(f"\n  GT 어노테이션: {gt_count}개", flush=True)
        print(f"  검출/GT 비율:  {sum(all_counts)}/{gt_count} = {sum(all_counts)/gt_count*100:.1f}%", flush=True)

    print(f"\n  시각화 저장: {OUTPUT_DIR}/", flush=True)
    print(f"\n  ✓ Stage 4 추론 완료", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())