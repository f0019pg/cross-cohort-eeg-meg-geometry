# Model-agnostic late neural geometry 및 brain-foundation bridge 사전고정 프로토콜 v001

작성일: 2026-08-07 KST  
상태: 새로운 CLIP/SigLIP adapter outcome 및 foundation-model embedding outcome을 계산하기 전에 고정  
목적: 양성 결과를 찾기 위한 사후 모델 선택이 아니라, 서로 다른 사전학습 원리를 대표하는 고정 분석군에서 기존 결과의 일반성과 새로운 neural-encoder 가능성을 계층적으로 검정한다.

## 1. 기존 결과와 새 분석의 구분

이미 열린 결과는 다음과 같다.

- DINOv3 기반 late EEG-MEG consensus adapter는 source EEG 및 MEG gate를 통과했다.
- THINGS-EEG2와 Alljoined에서는 양의 외부 이전을 보였다.
- NOD-EEG에서는 19/19가 양수였지만 사전고정 최소효과 0.005에 미달했다.
- 기존 static EEG teacher는 DINOv3, CLIP-L/14, SigLIP에서 분석된 적이 있으나, 현재의 late EEG-MEG consensus teacher는 DINOv3에서만 분석됐다.

따라서 Family A의 새 결과는 late consensus teacher의 backbone 일반성 검정이다. Family B와 C는 이전 원고 결과와 분리된 development 분석이며, NOD를 새로운 독립 confirmation으로 부르지 않는다.

모든 null, adverse, technical incompatibility를 삭제하지 않고 결과 파일에 보존한다. 결과를 본 뒤 backbone, layer, window, loss, padding, participant, class 또는 endpoint를 변경하지 않는다.

## 2. 분석군과 순서

1. Family A: parameter-matched multi-backbone late-consensus adapter
2. Family B0: neural foundation model의 체크포인트와 입력 호환성 감사
3. Family B1: frozen neural foundation embedding의 object 및 cross-modal 정보 검정
4. Family C: B1 gate를 통과한 경우에만 foundation-derived EEG-MEG geometry와 vision backbone의 융합

뒤 단계의 결과를 앞 단계의 모델 또는 threshold 선택에 사용하지 않는다.

## 3. Family A: multi-backbone late-consensus adapter

### 3.1 고정 backbone

- DINOv3 ViT-S/16, 384 dimensions: 기존 결과 재현용, 새로운 통계추론에 포함하지 않음
- CLIP ViT-B/32 image embedding, 512 dimensions: 새로운 검정 1
- SigLIP base image embedding, 768 dimensions: 새로운 검정 2

세 backbone 모두 동일한 Cichy 92-image 자극 순서로 추출된 최종 image embedding을 사용한다. `FEATURE_MANIFEST.json`의 고정된 72-image index를 사용하며, layer 선택은 하지 않는다.

### 3.2 고정 neural teacher와 교차적합

`late_crossmodal_consensus_distillation_20260806`의 다음 요소를 그대로 유지한다.

- Kaneshiro EEG 192-320 ms 후기 RDM
- Cichy MEG 180-300 ms 후기 RDM
- 홀수 참가자 teacher / 짝수 참가자 evaluation과 그 반대의 2 participant folds
- 범주마다 8개 train image와 4개 test image를 갖는 3 object folds
- EEG 및 MEG group RDM의 동일 가중 consensus
- seeds 20260722, 20260723, 20260724
- AdamW, learning rate 0.001, weight decay 0.0001, 400 epochs
- geometry anchor coefficient 100

### 3.3 Parameter matching

Adapter는 `LayerNorm -> down projection -> GELU -> up projection -> residual addition -> L2 normalization`이며 final projection은 zero initialization한다. DINOv3 384->64->384 adapter의 약 5만 trainable parameters에 맞추기 위해 다음 bottleneck을 고정한다.

- DINOv3: 64
- CLIP-B/32: 48
- SigLIP: 32

차이는 3% 이내이며 model dimension에 따른 adapter capacity 증가를 방지한다.

### 3.4 평가량

각 backbone에서 다음을 계산한다.

- held-out EEG alignment gain relative to its own frozen backbone
- held-out MEG alignment gain relative to its own frozen backbone
- adapter displacement와 DINO가 아니라 해당 frozen backbone 및 category-pair design을 제거한 후기 neural residual의 일치
- 세 object folds의 EEG 및 MEG gain
- adapted-frozen geometry preservation
- 범주 안 object label을 섞은 99개 teacher null

CLIP과 SigLIP의 참가자별 gain을 동일 가중 평균한 non-DINO family endpoint를 primary로 사용한다. 개별 backbone 결과는 해석 가능한 robustness component이며 Holm correction 전후 값을 모두 보고한다.

### 3.5 Family A gate

- A1 EEG family gain: 평균 > 0.005, 10명 중 8명 이상 양수, exact two-sided sign-flip P < 0.05
- A2 MEG family gain: 평균 > 0.005, 16명 중 12명 이상 양수, exact P < 0.05
- A3 direction: CLIP과 SigLIP 각각에서 EEG 및 MEG 평균 gain이 모두 양수
- A4 folds: 두 backbone 각각 세 object folds의 EEG와 MEG 평균 gain이 모두 양수
- A5 unique displacement: non-DINO family 평균이 EEG와 MEG 각각 > 0.02이고 exact P < 0.05
- A6 preservation: 모든 participant-fold x object-fold x backbone cell에서 adapted-frozen rho >= 0.95
- A7 specificity: jointly shuffled 99-teacher null에 대한 family-equal observed gain의 one-sided P < 0.05

모두 통과하면 `BACKBONE_GENERAL_LATE_CONSENSUS`다. A1-A2, A5-A7을 통과하지만 A3 또는 A4에서 한 backbone만 실패하면 `PARTIAL_BACKBONE_GENERALITY`다. 그 밖에는 `DINO_LIMITED_OR_INCONCLUSIVE`다.

## 4. Family B0: foundation-model compatibility audit

고정 후보와 우선순위는 다음과 같다.

1. BrainOmni-tiny: EEG와 MEG를 함께 지원하는 primary cross-modal model
2. REVE-Small 또는 공개된 최소 규모 checkpoint: EEG-only primary comparator
3. LaBraM-base: EEG-only comparator
4. CBraMod: EEG-only comparator

각 모델에 대해 공식 repository, checkpoint, license, hash, expected sampling rate, filter, duration, channel/montage requirements, validity-mask 지원 여부를 기록한다. 공식 checkpoint를 로드하고 dummy forward와 실제 NOD 1개 epoch forward가 성공해야 한다.

NOD 입력은 19명의 동일 참가자, 동일 4,000 images, 동일 1,000 ImageNet classes를 갖는 EEG 62 channels와 MEG 273 magnetometers다. 두 modality 모두 250 Hz, -0.1~0.8 s이며 sensor location이 존재한다.

Kaneshiro는 62.5 Hz, 1-25 Hz, 약 0.5 s로 이미 축소되어 있으므로 foundation-model primary input으로 사용하지 않는다. 현재 Cichy 파일은 trial-level sensor signal이 아니라 RDM이므로 foundation model에 직접 입력하지 않는다.

공식 모델이 0.9 s 입력 또는 validity mask를 지원하지 않으면 해당 모델은 `TECHNICALLY_INCOMPATIBLE`로 종료한다. 결과를 보기 위해 time tiling, reflection duplication 또는 임의 upsampling을 도입하지 않는다. Zero padding은 공식 attention mask가 padding을 완전히 배제할 때만 허용한다.

## 5. Family B1: frozen foundation embedding development endpoints

B0를 통과한 모델만 사용하며 backbone 전체를 fine-tune하지 않는다. 첫 분석은 frozen embedding, 두 번째 분석은 사전고정된 linear 또는 orthogonal head만 허용한다.

### 5.1 고정 분할

- class development/confirmation split: numeric class ID modulo 5가 0인 200 classes를 head development에 사용하고 나머지 800 classes를 once-only evaluation에 사용
- image split: 각 class의 정렬된 4개 image 중 위치 0,2는 half A, 1,3은 half B
- participant split: 홀수 participant teacher / 짝수 evaluation과 반대 방향을 평균
- hyperparameter selection은 200 development classes 안에서만 수행

### 5.2 Primary endpoints

- B1 class geometry reliability: half A와 half B로 만든 1,000-class RDM의 participant-level 일치
- B2 paired EEG-MEG geometry: 동일 참가자 EEG와 MEG class RDM의 일치
- B3 cross-modal class retrieval: EEG half A로 MEG half B의 class를 찾는 것과 반대 방향의 top-1, top-5, mean reciprocal rank
- B4 cross-participant retrieval: 다른 참가자 prototype에서 동일 class를 찾는 성능
- B5 pretrained specificity: random-initialized same architecture 및 covariance-matched PCA baseline 대비 향상

Image-label permutation 1,000회와 participant-level sign-flip을 사용한다. Foundation model이 참가자 수를 늘리는 것은 아니므로 population inference 단위는 참가자다.

BrainOmni가 B1-B5에서 방향 일관성을 보이고 B2 또는 B3가 permutation P < 0.01이며 19명 중 14명 이상 양수일 때만 Family C를 연다. EEG-only model은 cross-modal gate가 아니라 B1, B4 및 B5의 comparator로 사용한다.

## 6. Family C: gated neural-foundation fusion

Family C는 BrainOmni B1 gate가 통과한 경우에만 실행한다.

- visual backbones: DINOv3, CLIP-B/32, SigLIP의 동일 고정 panel
- neural target: participant-cross-fitted BrainOmni EEG-MEG class geometry
- train: 200 development classes와 image half A
- evaluation: 800 classes와 image half B, held-out participants
- vision backbone은 frozen, parameter-matched residual adapter만 학습
- primary outcome: frozen model 대비 held-out EEG 및 MEG foundation-embedding alignment gain
- secondary outcome: cross-modal class retrieval와 mean reciprocal rank의 변화
- controls: shuffled class teacher, random BrainOmni, EEG-only teacher, geometry preservation rho >= 0.95

이 단계는 NOD development 결과이며 새로운 독립 replication으로 주장하지 않는다. 성공할 경우 별도의 untouched EEG-MEG image dataset에서 confirmation이 필요하다.

## 7. 해석 경계

- Family A 성공: shared late neural teacher가 DINOv3에 특이적이지 않다는 robustness evidence
- Family B 성공: pretrained neural representation이 paired EEG-MEG object structure를 저표본 participant setting에서 더 안정적으로 드러낸다는 development evidence
- Family C 성공: foundation-derived joint EEG-MEG geometry가 여러 vision backbones에 transferable supervision을 제공한다는 candidate mechanism

어느 단계도 성공을 보장하지 않는다. Foundation model 사용만으로 biological sample size, 독립 replication 또는 causal mechanism이 증가하지 않는다.
