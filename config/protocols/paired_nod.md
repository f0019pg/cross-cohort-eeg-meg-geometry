# NOD 동일 참가자 EEG–MEG late-consensus adapter 이전 프로토콜 v001

고정일: 2026-08-07 KST  
상태: 현재 late-consensus adapter를 NOD-MEG에 적용한 결과를 계산하기 전 고정

## 1. 질문과 기존 분석과의 경계

Kaneshiro EEG와 Cichy MEG의 후기 물체 관계기하를 동일 가중치로 증류한 고정 residual adapter가, 학습에 사용되지 않은 NOD/ImageNet 이미지에 대해 **동일한 참가자의 후기 EEG와 후기 MEG 관계기하를 함께 더 잘 설명하는가**를 검정한다.

이 분석은 다음과 구분한다.

- `late_crossmodal_consensus_distillation_20260806`의 NOD-EEG 결과는 100–250 ms broadband endpoint였으며 이미 열렸다.
- `nod_meg_adapter_transfer_20260806`은 THINGS-EEG2 static teacher로 학습한 다른 adapter를 검정했다.
- `nod_image_disjoint_temporal_geometry_20260804`는 adapter가 아니라 NOD EEG early-to-late correction을 검정했다.

따라서 본 분석은 현재 late-consensus adapter의 **paired EEG–MEG late-window addendum**이다. NOD의 다른 neural outcomes가 앞선 분석에서 사용되었으므로 완전히 새로운 독립 confirmation으로 부르지 않고, prospectively specified cross-modal addendum으로 해석한다.

## 2. 고정 모델

- source teacher: Kaneshiro EEG 192–320 ms group RDM과 Cichy MEG 180–300 ms group RDM의 participant-rank-standardized 동일 가중치 평균
- backbone: frozen DINOv3 ViT-S/16, 384차원
- adapter: LayerNorm, 384→64→384, GELU, residual addition, output normalization
- checkpoints: `late_consensus_seed_20260722.pt`, `late_consensus_seed_20260723.pt`, `late_consensus_seed_20260724.pt`
- 세 seed의 adapted exact-image embedding을 평균한 뒤 단위 정규화한다.
- NOD 결과를 이용한 재학습, seed 선택, calibration, layer 선택 또는 parameter 조정은 하지 않는다.

## 3. 참가자와 이미지의 결과독립적 적격성

NOD EEG와 NOD MEG epoch 파일이 모두 존재하는 모든 참가자를 metadata-only audit 대상으로 한다. 참가자는 다음을 모두 만족할 때 포함한다.

1. EEG와 MEG에 공통으로 존재하고 DINOv3 feature가 있는 exact image가 1,000개 이상이다.
2. 공통 image가 ImageNet 1,000 classes 모두를 포함한다.
3. 각 class에 EEG와 MEG에서 공통인 image가 하나 이상 있다.
4. EEG와 MEG metadata의 공통 image에 대한 class ID가 일치한다.

각 참가자에서 공통 exact images만 사용한다. 한 class에 여러 공통 images가 있으면 해당 이미지들의 neural patterns와 model embeddings를 class 안에서 평균한다. 참가자와 class는 neural outcome을 보고 제외하지 않는다.

## 4. 고정 neural endpoint

- EEG 후기창: 192–320 ms
- MEG 후기창: 180–300 ms
- prestimulus: −100–0 ms
- poststimulus pattern은 각 trial의 −100–0 ms sensor mean을 뺀 뒤 sensors×time으로 펼친다.
- EEG는 모든 EEG channels, MEG는 모든 magnetometers를 사용한다.
- 동일 참가자·동일 modality·동일 class에서 공통 exact-image patterns를 평균한다.
- neural RDM은 class patterns 사이의 correlation distance이다.
- model RDM은 class-average embeddings 사이의 cosine distance이다.
- 참가자별 gain은 `Spearman(adapted RDM, neural RDM) − Spearman(frozen RDM, neural RDM)`이다.

## 5. 주 분석과 category sensitivity

### Native-bandwidth primary

NOD 저자가 공개한 0.1–100 Hz cleaned/epoched data를 그대로 사용한다. 이 결과는 preprocessing이 다른 외부 acquisition으로의 portability를 검정한다.

### Within-superclass sensitivity

NOD metadata의 고정 `super_class`가 같은 class pair만 사용해 동일 gain을 계산한다. 이를 통해 결과가 넓은 category 분리에만 의존하는지 평가한다.

## 6. 사전고정 bandwidth robustness

공개 epoch 전체에 25 Hz zero-phase low-pass를 적용한 뒤 동일 후기창 RDM을 다시 계산한다. 필터는 8차 Chebyshev type I, passband ripple 0.5 dB로 고정한다. 이 분석은 source EEG가 25 Hz low-pass였다는 bandwidth 차이에 대한 sensitivity이며 native 결과를 대체하지 않는다.

추가 low-pass의 acausal temporal smearing 가능성 때문에 low-pass prestimulus는 falsification gate로 사용하지 않는다. Native prestimulus만 시간적 음성대조로 사용한다. 25–100 Hz 단독 결과나 다른 cutoff는 결과를 본 뒤 추가하지 않는다.

## 7. 고정 통계와 gate

모든 참가자는 동일 가중치다. participant-level exact two-sided sign-flip test와 10,000회 participant bootstrap 95% CI를 사용한다. 적격 참가자가 19명이면 positive-count 기준은 아래와 같고, 다른 수가 되면 각각 `ceil(.75n)` 및 `ceil(.60n)`으로 자동 계산한다.

### Native core gates

- G1 EEG late: mean gain ≥ 0.005, ≥15/19 positive, exact P < 0.05
- G2 MEG late: mean gain ≥ 0.005, ≥15/19 positive, exact P < 0.05
- G3 paired joint: 참가자별 EEG/MEG gain 평균의 mean ≥ 0.005, exact P < 0.05, 두 modality 모두 positive인 참가자 ≥12/19
- G4 native temporal specificity: EEG와 MEG 각각 prestimulus absolute mean gain < 0.002 및 positive one-sided P ≥ 0.05. 각각 post-minus-pre mean > 0, ≥15/19 positive, exact P < 0.05
- G5 within-superclass: EEG와 MEG 각각 mean gain ≥ 0.003, ≥15/19 positive, exact P < 0.05
- G6 geometry preservation: 각 참가자 frozen–adapted RDM Spearman ρ의 EEG/MEG 평균이 각각 ≥0.95

### Bandwidth robustness gates

- B1 25-Hz EEG 후기 gain mean > 0, exact P < 0.05
- B2 25-Hz MEG 후기 gain mean > 0, exact P < 0.05
- B3 두 25-Hz modality gain이 모두 positive인 참가자 ≥12/19
- B4 25-Hz within-superclass gain이 EEG와 MEG 각각 mean > 0 및 exact P < 0.05

판정은 다음과 같다.

- native G1–G6와 B1–B4 모두 통과: `PAIRED_CROSSMODAL_TRANSFER_BANDWIDTH_ROBUST`
- native G1–G6만 통과: `PAIRED_CROSSMODAL_TRANSFER_NATIVE_ONLY`
- 그 외: `STOP_OR_LIMITED`

어떤 실패도 참가자, image, window, cutoff, distance, superclass, checkpoint 또는 threshold의 사후 교체를 유발하지 않는다.

## 8. 해석 경계

최강 판정이 나오면 다음 candidate statement가 허용된다.

> EEG와 MEG의 공통 후기 관계구조로 학습한 고정 adapter는 보지 않은 ImageNet 이미지에 대해 동일한 참가자들의 후기 EEG와 MEG 관계기하를 함께 더 잘 설명했으며, 이 방향은 25 Hz 이하로 대역을 맞춰도 유지되었다.

이 결과만으로 source localization, oscillatory mechanism, causal recurrence, 모든 주파수에서의 보편성, unseen class transfer 또는 완전히 독립적인 confirmation을 주장하지 않는다.
