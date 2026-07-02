# Binpicking Vision Pipeline

RTMDet-Ins 기반 2D 인스턴스 세그멘테이션과 ICP 3D 정합을 활용한 브라켓 빈피킹 시스템.

---

## 목차

1. [환경 요구사항](#1-환경-요구사항)
2. [설치](#2-설치)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [사용자 조정 변수](#4-사용자-조정-변수)
5. [실행 순서](#5-실행-순서)
6. [출력 파일 설명](#6-출력-파일-설명)
7. [TCP 통신 프로토콜](#7-tcp-통신-프로토콜)

---

## 1. 환경 요구사항

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| Python | 3.9 이상 |
| CUDA | 11.8 이상 (GPU 추론용) |
| GPU | NVIDIA (RTMDet CUDA 추론) |
| 카메라 | LUCID Helios ToF 카메라 |
| 로봇 | FAIRINO (TCP/IP 통신) |

---

## 2. 설치

### 2-1. Conda 환경 생성

```bash
conda create -n vision_env python=3.9 -y
conda activate vision_env
```

### 2-2. PyTorch 설치 (CUDA 버전에 맞게)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2-3. MMDetection 계열 설치

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
```

### 2-4. 기타 의존성 설치

```bash
pip install open3d opencv-python numpy pyyaml
```

### 2-5. LUCID 카메라 SDK 설치

```bash
# Arena SDK 설치 후
pip install arena_api
```

### 2-6. 프로젝트 설치

```bash
git clone <repo_url>
cd binpicking_vision/FINE_RTMDet
pip install -e .
```

---

## 3. 프로젝트 구조

```
FINE_RTMDet/
├── config/
│   └── config.yaml                        ← 카메라 설정
├── configs/
│   └── rtmdet-ins_bracket.py              ← 모델 학습 config (하이퍼파라미터)
├── data/
│   ├── cad/
│   │   └── bracket_v2.stl                 ← 브라켓 CAD 모델 (mm 단위)
│   ├── dataset/
│   │   └── YYYYMMDD_HHMMSS/               ← 학습 데이터 (1_Collect 출력)
│   │       ├── intensity/
│   │       ├── pointcloud_organized/
│   │       ├── valid_mask/
│   │       ├── metadata/
│   │       └── annotations/               ← 라벨링 후 추가
│   └── captures/
│       ├── binpicking.log                 ← 누적 실행 로그 ★
│       └── live/                          ← 실전 캡처 저장 경로
├── models/
│   └── rtmdet-ins_tiny_..._coco.pth       ← COCO 사전학습 가중치 (최초 학습용)
├── scripts/
│   ├── 0_1_Camera_capture_test.py         ← 카메라 연결 테스트
│   ├── 1_Collect_dataset.py               ← 학습 데이터 수집
│   ├── 2_Train_rtmdet_model.py            ← 모델 학습 (누적 학습 지원)
│   ├── 5_Run_binpicking.py                ← 실전 파이프라인 (TCP 서버)
│   └── TCP_client_test.py                 ← TCP 통신 테스트용 클라이언트
├── src/
│   ├── camera/                            ← 카메라 드라이버
│   └── detection/                         ← RTMDet inferencer
└── work_dirs/
    └── rtmdet-ins_bracket_v1/
        ├── rtmdet-ins_bracket.py          ← 학습에 사용된 config 사본
        ├── best_coco_bbox_mAP_epoch_N.pth ← 학습된 가중치 ★ (실전 사용)
        ├── epoch_N.pth                    ← 중간 체크포인트
        └── last_checkpoint                ← 마지막 체크포인트 포인터
```

---

## 4. 사용자 조정 변수

### 4-1. `config/config.yaml` — 카메라 설정

```yaml
camera:
  type: lucid_helios
  exposure_time_selector: Exp250Us
  operating_mode: Distance1500mm
  pixel_format: Coord3D_ABCY16
```

### 4-2. `5_Run_binpicking.py` — 실전 파이프라인 설정

**Detection**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SCORE_THRESHOLD` | `0.3` | 검출 신뢰도 임계값. 낮추면 더 많이 검출, 노이즈 증가 |
| `MIN_POINTS_PER_INSTANCE` | `100` | 이 이하 포인트 인스턴스는 노이즈로 제거 |
| `MASK_IOU_THRESHOLD` | `0.6` | 마스크 IoU 이 이상이면 중복 검출로 판단, score 낮은 것 제거 |

**ICP**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VOXEL_SIZE_CAD` | `0.002` (2mm) | CAD 다운샘플 크기. 작을수록 정밀하지만 느림 |
| `VOXEL_SIZE_SCENE` | `0.003` (3mm) | Scene 다운샘플 크기 |
| `ICP_FITNESS_THRESHOLD` | `0.5` | 이 값 미만이면 정합 실패 처리 |
| `ICP_STAGES` | `20→10→5mm` | 다단계 ICP 수렴 거리. 정합 실패 시 첫 단계 거리를 키움 |

**CAD 축 보정**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CAD_AXIS_CORRECTION_DEG` | `(-90, 90, 90)` | STL 좌표계 → 센서 좌표계 보정 회전각. CAD 교체 시 재확인 필요 |
| `CAD_PICK_LOCAL` | `[0.000, -0.100, 0.031, 1.0]` | CAD 로컬 좌표계에서 픽포인트 위치 (m 단위). CAD 교체 시 재측정 필요 |

**픽포인트 오프셋**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PICK_OFFSET_X_MM` | `-5.0` | 브라켓 폭 방향 오프셋 (mm) |
| `PICK_OFFSET_Y_MM` | `0.0` | 브라켓 길이 방향 오프셋 (mm) |
| `PICK_OFFSET_Z_MM` | `0.0` | 브라켓 높이 방향 오프셋 (mm) |

**TCP 서버**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `--port` | `29999` | TCP 포트 |
| `--host` | `0.0.0.0` | 바인드 주소 (변경 불필요) |

### 4-3. `2_Train_rtmdet_model.py` — 학습 설정

실행 인자로 조정.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | 없음 (필수) | 데이터셋 폴더명. 생략 시 목록에서 선택 |
| `--epochs` | config 값 사용 | 학습 epoch 수 override |

### 4-4. `1_Collect_dataset.py` — 데이터 수집 설정

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num` | `300` | 캡처할 프레임 수 |
| `--warmup` | `3` | 버리는 워밍업 프레임 수 |

---

## 5. 실행 순서

모든 명령은 프로젝트 루트에서 실행.

```bash
cd ~/binpicking_vision/FINE_RTMDet
conda activate vision_env
```

### Step 0. 카메라 연결 확인

처음 설치 후 또는 카메라 교체 시 실행.

```bash
python scripts/0_1_Camera_capture_test.py
```

확인 항목: 카메라 연결, 프레임 캡처, intensity PNG + PLY 저장, 통계 출력.

---

### Step 1. 학습 데이터 수집

```bash
python scripts/1_Collect_dataset.py --num 50
```

- 프레임마다 Enter를 눌러 부품 배치를 바꿔가며 촬영
- 출력 경로: `data/dataset/YYYYMMDD_HHMMSS/`
- 수집 완료 후 터미널에 다음 단계 명령어 자동 출력

---

### Step 2. RTMDet 모델 학습

수집한 데이터에 라벨링(CVAT 등) 후 실행.

```bash
# 데이터셋 폴더명 입력 방식
python scripts/2_Train_rtmdet_model.py --dataset 20260521_114500

# 인자 생략 시 → 목록에서 번호로 선택
python scripts/2_Train_rtmdet_model.py
```

**누적 학습 동작 방식**

```
최초 실행: models/rtmdet-ins_tiny_..._coco.pth 에서 시작
2차 이후:  work_dirs/.../best_*.pth 에서 자동으로 이어서 학습
```

학습 완료 후 `work_dirs/rtmdet-ins_bracket_v1/best_*.pth` 생성 확인.

---

### Step 3. 실전 실행 (TCP 서버)

카메라 + 로봇이 연결된 상태에서 실행.

```bash
python scripts/5_Run_binpicking.py
```

시작 시 출력 예시:
```
======================================================================
  빈피킹 TCP 서버
======================================================================
  Checkpoint:  best_coco_bbox_mAP_epoch_50.pth
  서버 IP:     192.168.0.15
  클라이언트 접속 주소: 192.168.0.15:29999
======================================================================
[4] TCP 대기 중: 192.168.0.15:29999
    클라이언트 연결을 기다립니다...
```

---

### Step 4. TCP 클라이언트 테스트 (로컬)

```bash
python scripts/TCP_client_test.py
```

```
Enter → 캡처 요청
수신: {'ok', 2, (-74.4, 3.3, 571.3, 2.35, 2.24, 1.11, 0.83), (214.7, 7.0, 567.7, ...)}
```

---

## 6. 출력 파일 설명

```
data/captures/
├── binpicking.log                         ← 누적 실행 로그 (서버 기동마다 구분선)
└── live/
    ├── intensity/
    │   └── YYYYMMDD_HHMMSS.png            ← 캡처 강도 이미지 (날짜시간으로 누적)
    ├── pointcloud_organized/
    │   └── frame_NNNN.npy                 ← organized PCD (H,W,3) mm
    ├── valid_mask/
    │   └── frame_NNNN.npy                 ← 유효 마스크 (H,W) bool
    ├── metadata/
    │   └── frame_NNNN.json                ← 캡처 통계
    └── results/
        ├── frame_NNNN_overlay.png         ← 2D detection 시각화 + 픽포인트 좌표 + ICP score
        ├── frame_NNNN_colored.ply         ← 통합 PCD (배경 회색 + 인스턴스별색 + CAD + 픽포인트)
        └── frame_NNNN_result.json         ← 모든 인스턴스 결과 통합
```

### `result.json` 구조

```json
{
  "frame": "frame_0001",
  "num_total": 2,
  "num_success": 2,
  "instances": [
    {
      "instance_id": 0,
      "icp_fitness": 0.83,
      "icp_rmse_m": 0.0012,
      "was_flipped": false,
      "pose": {
        "xyz_mm": [-74.4, 3.3, 571.3],
        "euler_deg": { "roll_deg": 2.35, "pitch_deg": 2.24, "yaw_deg": 1.11 },
        "transform_matrix": [[ ... ]]
      },
      "pick_point": {
        "position_mm": [-74.4, 3.3, 571.3],
        "approach_deg": { "roll_deg": 2.35, "pitch_deg": 2.24, "yaw_deg": 1.11 }
      }
    }
  ]
}
```

### PLY 색상 규칙 (Open3D 시각화)

| 색상 | 의미 |
|------|------|
| 회색 | 배경 포인트 클라우드 |
| 인스턴스 고유색 | 검출된 브라켓 포인트 |
| 초록 | ICP 정합된 CAD 모델 |
| 빨강 구 | 픽포인트 (그리퍼 목표 위치) |
| 파랑 선 | 접근 방향 벡터 |

### PLY 확인 명령

```bash
python -c "
import open3d as o3d
o3d.visualization.draw_geometries([
    o3d.io.read_point_cloud('data/captures/live/results/frame_0001_colored.ply')
])
"
```

---

## 7. TCP 통신 프로토콜

### 연결 방식

로봇(클라이언트) → 비전PC(서버), 한번 연결 후 반복 통신.

### 명령어

| 클라이언트 → 서버 | 의미 |
|---|---|
| `C\n` | 캡처 + 픽포인트 계산 요청 |
| `QUIT\n` | 서버 종료 |

### 응답 포맷

```
# 성공 (N개 검출)
{'ok', N, (x, y, z, roll, pitch, yaw, fit), (...), ...}

# 미검출 또는 ICP 전부 실패
{'No'}

# 오류
{'error', 'message'}
```

### 응답 필드 설명

| 필드 | 단위 | 설명 |
|------|------|------|
| x, y, z | mm | 픽포인트 위치 (카메라 좌표계) |
| roll, pitch, yaw | deg | 그리퍼 접근 각도 |
| fit | 0~1 | ICP 매칭 점수 (높을수록 정합 우수) |

### 페어이노 로봇 적용 예시

```python
import socket, ast

s = socket.socket()
s.connect(("192.168.0.15", 29999))

s.sendall(b"C\n")
raw = s.recv(4096).decode().strip()

# 응답 파싱
if raw.startswith("{'ok'"):
    inner  = raw.strip("{}")
    tokens = inner.split(", ", 2)
    count  = int(tokens[1])
    # 첫 번째 픽포인트 사용 (icp_fitness 가장 높은 순)
    # robot.MoveL([x, y, z, roll, pitch, yaw])
elif raw == "{'No'}":
    print("브라켓 없음")
```

### 로그 파일

```
data/captures/binpicking.log
```

서버 기동마다 날짜/시간 구분선이 자동으로 삽입되어 누적 저장됩니다.

```
======================================================================
  서버 기동: 2026-05-22 17:13:42
======================================================================
  ✓ 연결됨: ('172.30.1.7', 50832)
  수신: 'C'
  [frame_0001] 캡처 중...  2026-05-22 17:13:55
  검출: 2개  ICP: 성공 2개
  응답 전송: {'ok', 2, (-74.4, 3.3, 571.3, ...), (...)}
```