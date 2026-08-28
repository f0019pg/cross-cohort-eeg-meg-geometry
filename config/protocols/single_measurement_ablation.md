# Late EEG-MEG 3-teacher matched ablation 사전고정 프로토콜 v001

작성 시각: 2026-08-08 KST  
상태: matched-ablation outcome 계산 전 고정  
분류: prospectively locked post-hoc sensitivity analysis. 새 독립 확인이 아니다.

## 1. 질문

기존 분석은 독립 EEG와 MEG 집단에서 교차 재현되는 후기 물체 관계구조를 확인한 뒤, 두 측정의 동일 가중 consensus geometry를 frozen vision model의 residual adapter에 증류했다. 이 adapter는 held-out EEG와 MEG 정렬을 모두 개선했다.

본 분석은 다음의 남은 질문을 직접 검정한다.

> 동일한 참가자 분할, 이미지 분할, backbone, adapter, 최적화와 seed를 사용할 때 EEG-MEG consensus teacher가 EEG-only teacher 또는 MEG-only teacher보다 held-out 후기 신경 geometry에 더 유용한가.

기존 consensus 결과와 외부 결과는 이미 알려져 있다. 그러나 아래와 같은 완전 matched 3-teacher 결과는 아직 계산하지 않았다. 따라서 결과를 본 뒤 teacher 정의, endpoint, 방향, 분할 또는 판정 규칙을 바꾸지 않는다.

## 2. 고정 입력

- Kaneshiro EEG: 10명, 72개 이미지, 후기 192-320 ms
- Cichy MEG: 독립 16명, 동일한 72개 이미지, 후기 180-300 ms
- image backbone: 기존 DINOv3 ViT-S/16 384차원 feature
- category mapping: 기존에 고정한 6개 범주, 범주당 12개 이미지
- source code와 preprocessing: `late_crossmodal_consensus_distillation_20260806` Stage 1과 동일
- 비교 기준: frozen DINOv3

시간창, 이미지 순서, category mapping, neural RDM estimator, backbone 또는 exclusion은 바꾸지 않는다.

## 3. 동일한 participant와 object cross-fitting

Participant fold A에서는 홀수 인덱스 EEG 5명과 MEG 8명으로 teacher를 만들고 짝수 인덱스 EEG 5명과 MEG 8명을 평가한다. Participant fold B에서는 역할을 바꾼다. 각 평가 참가자는 같은 측정의 teacher에서 제외된다.

각 12-image 범주의 고정 순서를 세 부분으로 나누어 category-balanced 48 train 및 24 held-out image folds 세 개를 사용한다. 세 fold는 72개 이미지를 한 번씩 held-out 평가한다.

## 4. 세 teacher의 고정 정의

각 participant fold와 object fold에서 EEG와 MEG group RDM을 기존 코드와 동일하게 계산하고, training-image upper triangle을 rank-standardize한다.

- `EEG_ONLY`: EEG group late geometry
- `MEG_ONLY`: MEG group late geometry
- `CONSENSUS`: rank-standardized EEG와 MEG geometry의 0.5 대 0.5 평균을 다시 표준화한 값

Teacher 외의 모든 모델 요소는 동일하다. EEG-only와 MEG-only에 별도 scaling, reliability weighting, hyperparameter tuning 또는 modality-specific model을 사용하지 않는다.

## 5. 고정 adapter

- frozen DINOv3 ViT-S/16, width 384
- residual adapter: LayerNorm, 384-64-384, GELU, residual addition, output normalization
- zero-initialized final projection
- AdamW, learning rate 0.001, weight decay 0.0001
- anchor coefficient 100
- 400 epochs
- seeds 20260722, 20260723, 20260724
- 세 seed의 held-out embedding을 평균한 뒤 정규화

각 teacher는 2 participant folds x 3 object folds x 3 seeds의 동일한 18개 모델을 사용한다.

## 6. 평가량

각 teacher와 held-out participant에 대해 세 object folds의 값을 평균한다.

### 6.1 Frozen 대비 neural-alignment gain

Held-out 24개 이미지에서 adapted RDM과 participant late RDM의 Spearman correlation에서 frozen RDM과 participant late RDM의 correlation을 뺀다. EEG와 MEG에 각각 계산한다.

### 6.2 직접 matched contrasts

주요 비교는 다음 두 개다.

- `P1`: held-out EEG에서 `CONSENSUS minus EEG_ONLY`
- `P2`: held-out MEG에서 `CONSENSUS minus MEG_ONLY`

반대 측정에서의 portability도 모두 보고한다.

- held-out EEG에서 `CONSENSUS minus MEG_ONLY`
- held-out MEG에서 `CONSENSUS minus EEG_ONLY`
- held-out EEG와 MEG에서 `EEG_ONLY minus MEG_ONLY`

각 contrast는 participant-level mean, median, positive count, bootstrap mean 95% CI, exact one-sided positive sign-flip P와 exact two-sided sign-flip P를 보고한다. 세 object-fold별 mean contrast도 모두 보고한다.

### 6.3 균형 지표

Teacher별로 다음 두 값을 기술적으로 보고한다.

- macro gain: EEG mean gain과 MEG mean gain의 동일 가중 평균
- portability floor: EEG mean gain과 MEG mean gain 중 작은 값

두 측정의 표본 수가 다르므로 participant를 합쳐 하나의 P 값을 계산하지 않는다.

### 6.4 Geometry preservation

각 teacher의 adapted RDM과 frozen RDM의 Spearman correlation을 여섯 participant-fold x object-fold cell에서 계산하고 최솟값을 보고한다.

## 7. 결과 판정

### 기술적 유효성

- 세 teacher의 모든 54개 모델이 완료되어야 한다.
- 모든 participant x object-fold cell이 유한해야 한다.
- 새로 계산한 consensus participant gains가 기존 Stage 1 저장값과 최대 절대오차 1e-6 이내에서 일치해야 한다.
- 각 teacher의 최소 geometry preservation은 0.95 이상이어야 한다.

### `CONSENSUS_MATCHED_SUPERIORITY`

다음을 모두 만족할 때만 사용한다.

- P1 mean > 0, 8/10명 이상 positive, one-sided exact P < 0.05
- P2 mean > 0, 12/16명 이상 positive, one-sided exact P < 0.05
- P1과 P2의 세 object-fold mean이 모두 positive
- 기술적 유효성 통과

### `CONSENSUS_PORTABLE_NOT_MATCHED_SUPERIOR`

기술적 유효성과 기존 consensus frozen 대비 EEG 및 MEG 개선이 재현되지만 위 superiority gate를 모두 통과하지 못한 경우다. 이 경우 consensus가 양쪽 측정에 유용했다는 기존 주장은 유지할 수 있지만 single-measurement teacher보다 낫다고 주장하지 않는다.

### `SINGLE_TEACHER_SUFFICIENT_OR_PREFERRED`

P1 또는 P2의 평균이 0 이하이거나 해당 single teacher와 consensus 사이에 명확한 superiority 증거가 없고 single teacher의 macro gain 또는 portability floor가 consensus 이상이면 이 해석을 명시한다. 이 결과는 숨기거나 rescue하지 않는다.

### `TECHNICAL_FAILURE`

기술적 유효성 조건을 통과하지 못한 경우다. outcome과 무관한 code bug만 문서화하고 수정할 수 있다.

## 8. 주장 경계

이 분석은 기존 source data와 이미 열린 결과에 대한 post-hoc matched sensitivity analysis다. 통과하더라도 새 독립 confirmation, 일반적 modality superiority 또는 decoding 향상으로 주장하지 않는다. 실패하면 consensus가 EEG-only나 MEG-only보다 우수하다는 문장을 쓰지 않으며, 기존의 cross-measurement reproducibility와 consensus sufficiency만 제한적으로 보고한다.
