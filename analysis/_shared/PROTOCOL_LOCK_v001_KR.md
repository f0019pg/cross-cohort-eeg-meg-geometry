# 후기 EEG–MEG 공통 관계기하 증류 사전고정 프로토콜 v001

작성일: 2026-08-06 KST  
상태: 신경 결과를 새 통계량으로 계산하기 전에 고정

## 1. 질문과 기존 분석과의 구분

이 분석은 초기에서 후기로의 관계변화량을 교사로 사용하지 않는다. 주 질문은 다음과 같다.

> Kaneshiro EEG와 Cichy MEG의 후기 반응에 공통으로 존재하면서 DINOv3 시각 유사성과 대범주 구조만으로 설명되지 않는 물체 관계가 있는가. 그 공통 후기 구조를 고정된 시각모델의 작은 residual adapter에 증류하면 보지 않은 물체와 독립 신경자료에서도 유용한가.

기존 `crossmodal_temporal_correction_20260722`는 EEG의 early-to-late correction이 MEG late geometry를 예측하는지 검정했다. 기존 `build_kaneshiro_teacher_things_eval.py`는 Kaneshiro late EEG만을 교사로 사용했다. 본 분석은 두 방식과 다르게 EEG와 MEG에서 직접 교차 재현되는 **late geometry**만을 source gate로 삼는다.

## 2. 고정 자료와 시간창

- Kaneshiro EEG: 10명, 같은 72개 이미지, 6개 범주, 이미지당 72회 반복
- Cichy MEG: 독립된 16명, 위와 매핑된 같은 72개 이미지, 2개 session
- EEG 후기창: 192–320 ms
- MEG 후기창: 180–300 ms
- MEG 자극 전 대조창: −100–−1 ms
- DINOv3: 사전에 저장된 ViT-S/16 384차원 이미지 특징
- 범주: 매핑표에 고정된 6개 범주, 각 12개 이미지

시간창, 참가자, 이미지 매핑, 범주, backbone은 결과에 따라 변경하지 않는다.

## 3. 독립 추정과 참가자 분리

### EEG

각 참가자의 trial을 반복 순서에 따라 4개 fold로 나눈다. Fold 0×1과 fold 2×3의 cross-validated distance로 독립 후기 RDM 두 개를 만든다.

### MEG

각 참가자의 두 session에서 후기 RDM을 각각 만든다. 두 session은 반복측정 신뢰도 계산에 사용하며, 참가자별 후기 지도는 두 session의 rank-standardized 평균으로 만든다.

### 교차 modality 참가자 분리

- 16명 MEG group teacher로 10명의 EEG 참가자를 각각 평가한다.
- 10명 EEG group teacher로 16명의 MEG 참가자를 각각 평가한다.

두 cohort는 참가자가 완전히 다르므로 모든 평가 참가자는 자신이 포함되지 않은 반대 modality 교사로 평가된다. 임의의 홀짝 분할은 Stage 0에 사용하지 않는다. Adapter의 동일-modality source validation에만 사전고정된 홀짝 참가자 교차적합을 사용한다.

## 4. 시각 및 범주 통제

모든 pairwise distance를 rank 변환한다. 다음 변수를 선형 회귀한 잔차를 고유 후기 관계로 정의한다.

- DINOv3 image-pair distance
- 6개 범주의 모든 unordered category-pair type을 나타내는 dummy 변수

범주 내부 분석에서는 DINOv3 distance와 여섯 범주별 block intercept를 통제한다. 따라서 주 효과는 단순한 animate/inanimate 또는 범주 간 거리 차이로 설명될 수 없어야 한다.

## 5. Stage 0 source gate

### 신뢰도

- R1 EEG 후기 반복측정 신뢰도: 평균 ρ > 0.10, 10명 중 8명 이상 양수, exact two-sided sign-flip P < 0.05
- R2 MEG 후기 session 신뢰도: 평균 ρ > 0.02, 16명 중 12명 이상 양수, exact two-sided sign-flip P < 0.05

### 교차 modality 일반화

- S1 MEG 교사 → held-out EEG: 평균 partial Spearman ρ > 0.03, 10명 중 8명 이상 양수, exact P < 0.05
- S2 EEG 교사 → held-out MEG: 평균 partial Spearman ρ > 0.03, 16명 중 12명 이상 양수, exact P < 0.05
- S3 전체 EEG 교사와 전체 MEG 교사의 직접 잔차 상관 ρ > 0.05
- S4 modality에 같은 가중치를 준 평균 효과가 범주 내부 image-label permutation 9,999회에서 one-sided P < 0.05

### 반증 및 세밀한 관계

- S5 EEG 교사의 MEG 예측은 자극 전보다 후기에서 커야 한다. late-minus-pre 평균 > 0.02, 16명 중 12명 이상 양수, exact P < 0.05. 자극 전 양의 효과에 대한 one-sided sign-flip P는 0.05 이상이어야 한다.
- S6 범주 내부에서 MEG 교사 → EEG와 EEG 교사 → MEG가 각각 평균 ρ > 0.02, EEG 8/10 및 MEG 12/16 이상 양수, exact P < 0.05이어야 한다. 같은 통계량의 범주 내부 label permutation P < 0.05도 요구한다.

R1–R2와 S1–S6을 모두 통과할 때만 `GO_STAGE1_ADAPTER`로 판정한다. 하나라도 실패하면 `STOP_OR_LIMITED_SOURCE`이며 adapter를 열지 않는다.

## 6. Stage 1 adapter gate

Stage 0가 통과한 경우에만 구현하고 실행한다.

- 교사: 참가자 교차적합 fold별 EEG와 MEG 후기 group RDM의 동일 가중 평균
- 물체 교차검증: 각 범주의 고정 순서를 3등분하여 범주당 8개 학습 이미지와 4개 평가 이미지로 구성한 3 folds
- 모델: frozen DINOv3 ViT-S/16, 384→64→384 residual adapter, zero-initialized final projection
- seed: 20260722, 20260723, 20260724
- 평가: 학습에 사용하지 않은 24개 source 이미지와 교사에 포함되지 않은 EEG 및 MEG 참가자
- 필수 결과: frozen 대비 후기 EEG 및 MEG alignment gain, adapter-induced relational displacement와 시각·범주 통제 후기 잔차의 상관, frozen geometry 보존
- specificity: 범주 내부 teacher-label shuffle 39회, 고정 seed 20260722

정확한 수치 gate와 loss coefficient는 Stage 0 결과를 열기 전에 별도 Amendment로 고정한다. Stage 0 결과를 보고 값을 선택하지 않는다.

## 7. Stage 2 외부 평가

Stage 1이 통과한 경우에만 실행한다.

- THINGS confirmation 183 concepts와 fitting에 쓰이지 않은 photograph half
- THINGS-EEG2 참가자
- Alljoined 참가자
- 독립 human similarity judgement
- NOD-EEG의 ImageNet images와 참가자

최종 source adapter는 source 참가자와 72개 이미지만 사용해 학습하며, 외부 평가에는 재학습하지 않는다. 이전에 살펴본 외부 자료이므로 이 단계는 독립 confirmatory test가 아니라 exploratory candidate evidence로 명시한다.

## 8. 해석 경계

통과하더라도 주장할 수 있는 것은 다음 범위다.

- 후기 물체 관계 중 일부가 독립 참가자와 EEG/MEG 측정을 넘어 재현된다.
- 그 교차 modality 후기 구조가 작은 frozen-backbone adapter의 유용한 교사가 될 수 있다.

source localization, 특정 주파수 기전, recurrence, 인과성, 보편적 의미표상은 주장하지 않는다. 실패 또는 adverse 결과도 그대로 보존한다.
