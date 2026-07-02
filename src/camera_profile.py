"""

카메라를 바꿀 때는 이 파일의 CAMERA_PROFILE 값만 수정하면
0_Camera_capture_test.py / 1_Collect_dataset.py /
6_Run_binpicking_TCP_UI.py 등 전부에 자동으로 반영됩니다.
값은 config/config_{CAMERA_PROFILE}.yaml 파일명과 정확히 일치해야 함.

"""

# "lucid_helios" | "femto_bolt"
CAMERA_PROFILE = "femto_bolt"