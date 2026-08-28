# Amendment 02. Stage 2 외부 이전 분석 고정

작성 시점: Stage 1이 `GO_STAGE2_EXTERNAL`로 종료된 뒤, 새 late-consensus adapter로 THINGS, Alljoined 또는 NOD outcome을 계산하기 전

## 1. 목적

Stage 1은 late EEG–MEG consensus adapter가 같은 72-image source set 안에서 보지 않은 이미지와 참가자로 일반화됨을 보였다. Stage 2는 모든 source 참가자와 72개 이미지만 사용해 최종 adapter를 학습한 뒤, 재학습 없이 독립 object concepts, photographs, EEG acquisitions와 ImageNet image database로 옮겨지는지 검정한다.

이 외부 데이터들은 프로젝트의 이전 분석에서 이미 사용되었으므로 본 단계는 새로운 독립 confirmation이 아니라 prospectively specified exploratory external validation으로 해석한다.

## 2. 최종 source teacher와 adapter

- Kaneshiro EEG 10명의 후기 RDM을 participant-rank standardization 후 평균한다.
- Cichy MEG 16명의 후기 RDM을 같은 방식으로 평균한다.
- 두 group maps를 각각 standardize하고 동일 가중치로 평균한 72-image late consensus를 target으로 사용한다.
- DINOv3 backbone, 64-unit residual adapter, zero initialization, 400 epochs, AdamW, learning rate 0.001, weight decay 0.0001, anchor 100을 Stage 1과 동일하게 유지한다.
- Seeds: 20260722, 20260723, 20260724
- Source 외부에서는 어떤 parameter도 다시 맞추지 않는다.

## 3. THINGS와 Alljoined 평가

- Frozen split의 confirmation 183 concepts만 사용한다.
- 각 concept의 evaluation photograph half B를 사용한다.
- THINGS-EEG2 8명과 Alljoined 20명의 고정 EEG half 1을 평가한다.
- 참가자별 endpoint는 adapted RDM과 EEG RDM의 Spearman correlation에서 frozen DINOv3 값을 뺀 gain이다.
- 독립 SPoSE human similarity judgement와의 correlation 및 gain은 convergent descriptive endpoint로 모두 보고하지만 neural transfer gate에는 포함하지 않는다.

## 4. NOD/ImageNet 평가

- 기존 감사가 끝난 55,761개 DINOv3 image features와 고정 trial index를 사용한다.
- 참가자별 1,000 ImageNet class에 대해 trial image features를 평균하고 class RDM을 만든다.
- EEG endpoint는 100–250 ms poststimulus class RDM이다.
- −100–0 ms class RDM은 prestimulus falsification이다.
- 19명 각각에서 adapted-minus-frozen alignment gain을 계산한다.

## 5. Seeds와 ensemble

각 image embedding에 세 seed adapter를 적용하고 embedding을 평균한 뒤 단위 정규화한다. Ensemble RDM을 frozen RDM 및 neural RDM과 비교한다. NOD에서도 class averaging 전에 세 seed의 adapted image embedding을 평균한다.

## 6. 외부 gates

- E1 THINGS-EEG2: mean gain ≥ 0.005, 8/8 positive, exact two-sided sign-flip P < 0.05
- E2 Alljoined: mean gain ≥ 0.005, ≥15/20 positive, exact P < 0.05
- E3 NOD poststimulus: mean gain ≥ 0.005, ≥15/19 positive, exact P < 0.05
- E4 NOD temporal specificity: absolute prestimulus mean gain < 0.005, positive one-sided sign-flip P ≥ 0.05, post-minus-pre mean > 0.005, ≥15/19 positive, exact two-sided P < 0.05
- E5 geometry preservation: THINGS confirmation RDM ρ ≥ 0.95, NOD participant-average ρ ≥ 0.95

E1–E5가 모두 통과할 때 `EXTERNAL_TRANSFER_CANDIDATE`이다. 하나라도 실패하면 `SOURCE_SUPPORTED_EXTERNAL_LIMITED`이며 실패 결과를 그대로 보존한다.

## 7. 해석 경계

통과하면 다음을 candidate finding으로 기술할 수 있다.

> EEG와 MEG에 공통인 후기 물체 관계를 72개 source images에서 학습한 작은 adapter가, 재학습 없이 새로운 concepts와 photographs의 독립 EEG 및 다른 ImageNet image database의 EEG 정렬을 개선했다.

이는 새 참가자나 새 데이터에서 사전에 봉인된 독립 replication은 아니며, source analysis와 기존에 사용한 외부 자산을 연결한 exploratory validation이다. 독립 confirmation이 이루어지기 전에는 확정적 discovery 또는 universal late code로 표현하지 않는다.
