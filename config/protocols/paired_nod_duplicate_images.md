# Amendment 02. sub-05 EEG duplicate image metadata 처리

작성 시점: sub-01–04 cache가 완료되고 sub-05의 metadata alignment에서 기술적으로 중단된 뒤, sub-05 또는 이후 참가자의 neural 결과가 계산되기 전

Metadata-only 진단에서 다음을 확인했다.

- sub-05 EEG: 4,000 trial rows, 3,761 unique image IDs, 239 duplicate image occurrences
- sub-05 MEG: 4,000 trial rows, 4,000 unique image IDs
- EEG와 MEG에 공통이고 feature가 있는 unique images: 3,761
- 공통 image의 class ID는 두 modality에서 모두 일치하며 1,000 classes를 전부 포함한다.

원 코드가 image ID를 row index로 직접 변환하면서 sub-05 EEG의 중복 occurrences를 4,000 rows로 확장해 MEG의 3,761 rows와 길이가 달라졌다. 이는 neural outcome과 무관한 metadata indexing bug다.

수정은 다음과 같이 고정한다.

1. 각 modality에서 공통 unique image ID에 속하는 모든 trial rows를 선택한다.
2. 동일 image ID가 반복되면 sensor×time pattern을 먼저 image 안에서 평균한다.
3. 그 뒤 공통 unique images를 class 안에서 평균한다.
4. model embedding은 각 공통 unique image당 한 번 사용한다.

따라서 모든 참가자에서 exact images가 동일 가중치가 되며, sub-01–04처럼 image ID가 고유한 참가자에서는 기존 계산과 수치적으로 동일하다. 참가자 적격성, 시간창, 대역, 모델, endpoints 및 gates는 변경하지 않는다.
