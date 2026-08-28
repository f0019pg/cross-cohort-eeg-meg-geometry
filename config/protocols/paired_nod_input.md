# Amendment 01. 공개 NOD epoch의 native bandwidth 정정

작성 시점: metadata-only `--check-inputs`가 완료된 뒤, 현재 late-consensus adapter를 NOD-MEG neural data에 적용하기 전

원 프로토콜은 NOD 논문에 기술된 cleaned time series의 범위를 따라 native data를 0.1–100 Hz로 기술했다. 그러나 실제 분석에 사용할 공개 epoched FIF header를 metadata-only로 감사한 결과, 19명의 EEG와 MEG epoch 모두 다음과 같았다.

- high-pass: 0.1 Hz
- low-pass: 40 Hz
- sampling frequency: 250 Hz

따라서 본 분석의 native endpoint는 **공개 epoch에 저장된 0.1–40 Hz data**로 정정한다. 25 Hz low-pass sensitivity, 시간창, 참가자 적격성, 모델, endpoints, gates 및 해석 경계는 변경하지 않는다. 이 정정은 neural values를 읽기 전에 입력 메타데이터만으로 이루어졌다.
